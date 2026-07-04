import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session

from auth.models import User
from auth.security import hash_password, verify_password, create_access_token, decode_token
from auth.security_middleware import limiter, verify_turnstile, is_turnstile_enabled, TURNSTILE_SITE_KEY
from services.email_service import send_email

logger = logging.getLogger(__name__)

VERIFICATION_CODE_TTL_MIN = 15
PASSWORD_RESET_TTL_HOURS = 1
_email_jinja = Environment(loader=FileSystemLoader("templates"), autoescape=True)


def _generate_code() -> str:
    """6-значный код, ведущие нули сохраняются."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _send_verification_email(user: User, code: str) -> bool:
    tmpl = _email_jinja.get_template("email_verification.html")
    html = tmpl.render(
        username=user.username, code=code,
        expires_minutes=VERIFICATION_CODE_TTL_MIN,
    )
    return send_email(user.email, "Trading Signals · код подтверждения", html)


def _send_reset_email(user: User, reset_url: str) -> bool:
    tmpl = _email_jinja.get_template("email_password_reset.html")
    html = tmpl.render(
        username=user.username, reset_url=reset_url,
        expires_hours=PASSWORD_RESET_TTL_HOURS,
    )
    return send_email(user.email, "Trading Signals · восстановление пароля", html)

router = APIRouter(prefix="/auth", tags=["Auth"])
templates = Jinja2Templates(directory="templates")

COOKIE_NAME = "access_token"


def get_current_user(request: Request, db: Session) -> Optional[User]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    username = payload.get("sub")
    if not username:
        return None
    return db.query(User).filter(User.username == username, User.is_active == True).first()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, reset: str = ""):
    info = "Пароль успешно обновлён. Войдите с новым паролем." if reset else None
    return templates.TemplateResponse(
        request=request, name="login.html",
        context={"error": None, "info": info, "form": {}, "turnstile_site_key": TURNSTILE_SITE_KEY},
    )


@router.post("/login", response_class=HTMLResponse)
@limiter.limit("5/5minute")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    from main import get_db_session
    db = get_db_session()
    try:
        form_data = {"username": username}
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "error": "Неверное имя пользователя или пароль",
                    "form": form_data, "info": None,
                    "turnstile_site_key": TURNSTILE_SITE_KEY,
                },
                status_code=401,
            )
        if not user.is_active:
            # Дадим пользователю прямой путь дальше — подтвердить email
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "error": (
                        f"Аккаунт не подтверждён. "
                        f"<a href='/auth/verify?email={user.email}'>Ввести код с почты</a>."
                    ),
                    "error_is_html": True,
                    "form": form_data, "info": None,
                    "turnstile_site_key": TURNSTILE_SITE_KEY,
                },
                status_code=403,
            )
        user.last_login = datetime.utcnow()
        db.commit()

        token = create_access_token({"sub": user.username})
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            COOKIE_NAME, token,
            httponly=True,
            max_age=60 * 60 * 24 * 7,
            samesite="lax",
            secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        )
        return response
    finally:
        db.close()


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="register.html",
        context={"error": None, "form": {}, "turnstile_site_key": TURNSTILE_SITE_KEY},
    )


@router.post("/register", response_class=HTMLResponse)
@limiter.limit("10/hour")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(""),
    password: str = Form(...),
    password2: str = Form(...),
    cf_turnstile_response: str = Form(default=""),
):
    form_data = {"username": username, "email": email, "full_name": full_name}

    # Проверка CAPTCHA (если настроена)
    if is_turnstile_enabled():
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
                    or request.client.host
        if not verify_turnstile(cf_turnstile_response, client_ip):
            return templates.TemplateResponse(
                request=request, name="register.html",
                context={
                    "error": "Не удалось проверить капчу. Попробуйте ещё раз.",
                    "form": form_data,
                    "turnstile_site_key": TURNSTILE_SITE_KEY,
                },
                status_code=400,
            )

    from main import get_db_session
    db = get_db_session()
    try:
        def _err(msg: str):
            return templates.TemplateResponse(
                request=request,
                name="register.html",
                context={
                    "error": msg,
                    "form": form_data,
                    "turnstile_site_key": TURNSTILE_SITE_KEY,
                },
            )
        if password != password2:
            return _err("Пароли не совпадают")
        if len(password) < 6:
            return _err("Пароль должен быть не менее 6 символов")
        if db.query(User).filter(User.username == username).first():
            return _err("Имя пользователя уже занято")
        if db.query(User).filter(User.email == email).first():
            return _err("Email уже используется")

        # Юзер создаётся неактивным до подтверждения email
        code = _generate_code()
        user = User(
            username=username,
            email=email,
            full_name=full_name or None,
            hashed_password=hash_password(password),
            is_active=False,
            verification_code=code,
            verification_expires_at=datetime.utcnow() + timedelta(minutes=VERIFICATION_CODE_TTL_MIN),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        if not _send_verification_email(user, code):
            logger.error(f"Не удалось отправить код на {email}")
            return _err("Не удалось отправить письмо с кодом. Попробуй позже.")

        return RedirectResponse(url=f"/auth/verify?email={email}", status_code=302)
    finally:
        db.close()


# ============================================================
# Email verification
# ============================================================

@router.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request, email: str = ""):
    return templates.TemplateResponse(
        request=request, name="verify.html",
        context={"email": email, "error": None, "info": None},
    )


@router.post("/verify", response_class=HTMLResponse)
@limiter.limit("10/hour")
async def verify(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
):
    from main import get_db_session
    db = get_db_session()
    try:
        def _err(msg: str):
            return templates.TemplateResponse(
                request=request, name="verify.html",
                context={"email": email, "error": msg, "info": None},
            )
        code = code.strip()
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return _err("Пользователь не найден")
        if user.is_active and user.email_verified_at:
            return _err("Аккаунт уже подтверждён — войдите в личный кабинет")
        if not user.verification_code or not user.verification_expires_at:
            return _err("Код не запрошен. Получи новый через кнопку ниже.")
        if datetime.utcnow() > user.verification_expires_at:
            return _err("Код устарел. Получи новый через кнопку ниже.")
        if code != user.verification_code:
            return _err("Неверный код")

        user.is_active = True
        user.email_verified_at = datetime.utcnow()
        user.verification_code = None
        user.verification_expires_at = None
        db.commit()

        token = create_access_token({"sub": user.username})
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            COOKIE_NAME, token, httponly=True,
            max_age=60 * 60 * 24 * 7, samesite="lax",
            secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        )
        return response
    finally:
        db.close()


@router.post("/resend-verification")
@limiter.limit("10/hour")
async def resend_verification(request: Request, email: str = Form(...)):
    from main import get_db_session
    db = get_db_session()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and not user.is_active:
            code = _generate_code()
            user.verification_code = code
            user.verification_expires_at = datetime.utcnow() + timedelta(minutes=VERIFICATION_CODE_TTL_MIN)
            db.commit()
            _send_verification_email(user, code)
        # Не раскрываем существует ли email — редиректим одинаково
        return RedirectResponse(
            url=f"/auth/verify?email={email}", status_code=302,
        )
    finally:
        db.close()


# ============================================================
# Password reset
# ============================================================

@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="forgot_password.html",
        context={"info": None, "form": {}},
    )


@router.post("/forgot-password", response_class=HTMLResponse)
@limiter.limit("10/hour")
async def forgot_password(request: Request, email: str = Form(...)):
    from main import get_db_session
    db = get_db_session()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and user.is_active:
            token = secrets.token_urlsafe(48)[:64]
            user.password_reset_token = token
            user.password_reset_expires_at = datetime.utcnow() + timedelta(hours=PASSWORD_RESET_TTL_HOURS)
            db.commit()
            base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8002").rstrip("/")
            _send_reset_email(user, f"{base_url}/auth/reset-password?token={token}")
        # Показываем одинаковое сообщение независимо от того, есть ли email
        return templates.TemplateResponse(
            request=request, name="forgot_password.html",
            context={
                "info": "Если такой email зарегистрирован, ссылка для сброса пароля отправлена.",
                "form": {},
            },
        )
    finally:
        db.close()


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = ""):
    from main import get_db_session
    db = get_db_session()
    try:
        user = db.query(User).filter(User.password_reset_token == token).first() if token else None
        if not user or not user.password_reset_expires_at or datetime.utcnow() > user.password_reset_expires_at:
            return templates.TemplateResponse(
                request=request, name="reset_password.html",
                context={"token": None, "error": "Ссылка недействительна или устарела. Запросите новую."},
            )
        return templates.TemplateResponse(
            request=request, name="reset_password.html",
            context={"token": token, "error": None},
        )
    finally:
        db.close()


@router.post("/reset-password", response_class=HTMLResponse)
@limiter.limit("5/hour")
async def reset_password(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    from main import get_db_session
    db = get_db_session()
    try:
        def _err(msg: str, tok=token):
            return templates.TemplateResponse(
                request=request, name="reset_password.html",
                context={"token": tok, "error": msg},
            )
        if password != password2:
            return _err("Пароли не совпадают")
        if len(password) < 6:
            return _err("Пароль должен быть не менее 6 символов")
        user = db.query(User).filter(User.password_reset_token == token).first()
        if not user or not user.password_reset_expires_at or datetime.utcnow() > user.password_reset_expires_at:
            return _err("Ссылка недействительна или устарела", tok=None)

        user.hashed_password = hash_password(password)
        user.password_reset_token = None
        user.password_reset_expires_at = None
        db.commit()
        return RedirectResponse(url="/auth/login?reset=1", status_code=302)
    finally:
        db.close()


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/auth/login", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    from main import get_db_session
    db = get_db_session()
    try:
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)
        return templates.TemplateResponse(
            request=request, name="profile.html", context={"user": user}
        )
    finally:
        db.close()


@router.post("/profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    full_name: str = Form(""),
    email: str = Form(...),
    email_subscribed: str = Form(""),
):
    from main import get_db_session
    db = get_db_session()
    try:
        user = get_current_user(request, db)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)

        existing = db.query(User).filter(User.email == email, User.id != user.id).first()
        if existing:
            return templates.TemplateResponse(
                request=request,
                name="profile.html",
                context={"user": user, "error": "Email уже используется"},
            )
        user.full_name = full_name or None
        user.email = email
        user.email_subscribed = bool(email_subscribed)
        db.commit()
        db.refresh(user)
        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={"user": user, "success": "Профиль обновлён"},
        )
    finally:
        db.close()


@router.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe(request: Request, token: str = ""):
    """
    Отписка от рассылки по ссылке из футера письма.
    Токен уникальный на юзера, логин не требуется.
    """
    from main import get_db_session
    db = get_db_session()
    try:
        if not token:
            return HTMLResponse("Некорректная ссылка отписки.", status_code=400)
        user = db.query(User).filter(User.unsubscribe_token == token).first()
        if not user:
            return HTMLResponse(
                "Ссылка недействительна или устарела. Отпишитесь в личном кабинете.",
                status_code=404,
            )
        user.email_subscribed = False
        db.commit()
        return HTMLResponse(f"""
            <html><body style="font-family: sans-serif; max-width: 480px; margin: 80px auto; text-align: center;">
                <h2>Отписка оформлена</h2>
                <p>Больше не будем присылать письма на {user.email}.</p>
                <p><a href="/">На главную</a></p>
            </body></html>
        """)
    finally:
        db.close()
