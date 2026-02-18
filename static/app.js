/**
 * Скрипт для страницы результатов моделей прогнозирования
 */

// Инициализация страницы
window.onload = function() {
    highlightSignals();
    updateTime();
    setupSortingListeners();
};

// Функция для подсветки торговых сигналов в тексте
function highlightSignals() {
    const cardBodies = document.querySelectorAll('.card-body');

    cardBodies.forEach(body => {
        const html = body.innerHTML;

        // Подсветка различных торговых сигналов
        let processedHtml = html
            .replace(/(Торговый сигнал: BUY)/g, '<span class="signal-highlight" style="color: green;">$1</span>')
            .replace(/(Торговый сигнал: SELL)/g, '<span class="signal-highlight" style="color: red;">$1</span>')
            .replace(/(Торговый сигнал: HOLD)/g, '<span class="signal-highlight" style="color: orange;">$1</span>')
            .replace(/(Торговый сигнал: NEUTRAL)/g, '<span class="signal-highlight" style="color: gray;">$1</span>');

        // Подсветка показателей точности и ошибок
        processedHtml = processedHtml
            .replace(/(Точность направления: \d+\.\d+%)/g, '<strong>$1</strong>')
            .replace(/(Direction Accuracy: \d+\.\d+)/g, '<strong>$1</strong>')
            .replace(/(MAPE: \d+\.\d+%)/g, '<strong>$1</strong>')
            .replace(/(Ошибка: .*?%)/g, '<span style="color: red;">$1</span>')
            .replace(/(R²: \d+\.\d+)/g, '<strong>$1</strong>');

        // Подсветка цен и предсказаний
        processedHtml = processedHtml
            .replace(/(Текущая цена: \d+\.\d+)/g, '<strong>$1</strong>')
            .replace(/(Прогнозируемая цена: \d+\.\d+)/g, '<strong>$1</strong>')
            .replace(/(Ожидаемое изменение: [+-]?\d+\.\d+%)/g, '<strong>$1</strong>');

        body.innerHTML = processedHtml;
    });
}

// Функция для обновления текущего времени
function updateTime() {
    const now = new Date();
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    const timeString = now.toLocaleString(undefined, options);
    const timeInfo = document.getElementById('time-info');
    if (timeInfo) {
        timeInfo.textContent = `Последнее обновление: ${timeString}`;
    }
}

// Настройка слушателей событий для кнопок сортировки
function setupSortingListeners() {
    const sortingOptions = document.querySelectorAll('.sorting-option');

    sortingOptions.forEach(option => {
        option.addEventListener('click', function() {
            const field = this.getAttribute('data-sort');

            // Определяем текущее направление сортировки
            let direction = 'desc'; // По умолчанию по убыванию

            // Если на этом элементе уже есть активный класс, меняем направление
            if (this.classList.contains('active')) {
                const arrow = this.querySelector('.sort-arrow');
                if (arrow && arrow.classList.contains('desc')) {
                    direction = 'asc';
                } else {
                    direction = 'desc';
                }
            }

            // Обновляем UI - убираем активный класс со всех кнопок сортировки
            sortingOptions.forEach(btn => {
                btn.classList.remove('active');
                const btnArrow = btn.querySelector('.sort-arrow');
                if (btnArrow) {
                    btnArrow.classList.remove('asc', 'desc');
                }
            });

            // Активируем текущую кнопку и стрелку
            this.classList.add('active');
            const currentArrow = this.querySelector('.sort-arrow');
            if (currentArrow) {
                currentArrow.classList.add(direction);
            }

            // Выполняем сортировку
            sortCards(field, direction);
        });
    });
}

// Функция сортировки карточек
function sortCards(field, direction) {
    const container = document.querySelector('.container');
    const cards = Array.from(container.querySelectorAll('.card'));

    // Функция для получения значения по полю
    const getValue = (card, field) => {
        switch(field) {
            case 'signal':
                return card.getAttribute('data-signal') || '';
            case 'accuracy':
                return parseFloat(card.getAttribute('data-accuracy') || 0);
            case 'ticker':
                return card.getAttribute('data-ticker') || '';
            case 'date':
                return card.getAttribute('data-date') || '';
            case 'expected_change':
                return parseFloat(card.getAttribute('data-expected-change') || 0);
            case 'r_squared':
                return parseFloat(card.getAttribute('data-r-squared') || 0);
            default:
                return 0;
        }
    };

    // Сортировка карточек
    cards.sort((a, b) => {
        const valueA = getValue(a, field);
        const valueB = getValue(b, field);

        // Обработка случая, когда значения равны null, undefined или NaN
        if (!valueA && valueA !== 0) return direction === 'asc' ? -1 : 1;
        if (!valueB && valueB !== 0) return direction === 'asc' ? 1 : -1;

        // Для числовых значений
        if (typeof valueA === 'number' && typeof valueB === 'number') {
            return direction === 'asc' ? valueA - valueB : valueB - valueA;
        }

        // Для строковых значений
        return direction === 'asc'
            ? String(valueA).localeCompare(String(valueB))
            : String(valueB).localeCompare(String(valueA));
    });

    // Добавляем отсортированные карточки обратно в контейнер
    cards.forEach(card => container.appendChild(card));
}