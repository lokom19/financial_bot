import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from auth.models import User
from auth.security import hash_password, verify_password, create_access_token, decode_token
from auth.security_middleware import limiter, verify_turnstile, is_turnstile_enabled, TURNSTILE_SITE_KEY

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
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html",
        context={"error": None, "turnstile_site_key": TURNSTILE_SITE_KEY},
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
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"error": "Неверное имя пользователя или пароль"},
                status_code=401,
            )
        if not user.is_active:
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"error": "Аккаунт заблокирован"},
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
        context={"error": None, "turnstile_site_key": TURNSTILE_SITE_KEY},
    )


@router.post("/register", response_class=HTMLResponse)
@limiter.limit("3/hour")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(""),
    password: str = Form(...),
    password2: str = Form(...),
    cf_turnstile_response: str = Form(default=""),
):
    # Проверка CAPTCHA (если настроена)
    if is_turnstile_enabled():
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
                    or request.client.host
        if not verify_turnstile(cf_turnstile_response, client_ip):
            return templates.TemplateResponse(
                request=request, name="register.html",
                context={
                    "error": "Не удалось проверить капчу. Попробуйте ещё раз.",
                    "turnstile_site_key": TURNSTILE_SITE_KEY,
                },
                status_code=400,
            )

    from main import get_db_session
    db = get_db_session()
    try:
        # Хелпер чтобы каждый раз не передавать turnstile_site_key вручную
        def _err(msg: str):
            return templates.TemplateResponse(
                request=request,
                name="register.html",
                context={"error": msg, "turnstile_site_key": TURNSTILE_SITE_KEY},
            )
        if password != password2:
            return _err("Пароли не совпадают")
        if len(password) < 6:
            return _err("Пароль должен быть не менее 6 символов")
        if db.query(User).filter(User.username == username).first():
            return _err("Имя пользователя уже занято")
        if db.query(User).filter(User.email == email).first():
            return _err("Email уже используется")

        user = User(
            username=username,
            email=email,
            full_name=full_name or None,
            hashed_password=hash_password(password),
        )
        db.add(user)
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
