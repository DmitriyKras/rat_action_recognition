#!/bin/bash

# Проверяем, установлен ли ffprobe (часть ffmpeg)
if ! command -v ffprobe &> /dev/null; then
    echo "Ошибка: ffprobe не установлен. Установите ffmpeg для работы скрипта."
    echo "Для Ubuntu/Debian: sudo apt install ffmpeg"
    echo "Для CentOS/RHEL: sudo yum install ffmpeg"
    echo "Для macOS: brew install ffmpeg"
    exit 1
fi

# Проверяем, что указана директория
if [ -z "$1" ]; then
    echo "Использование: $0 <путь_к_директории>"
    echo "Пример: $0 /path/to/videos"
    exit 1
fi

target_dir="$1"

# Проверяем, что директория существует
if [ ! -d "$target_dir" ]; then
    echo "Ошибка: директория '$target_dir' не существует"
    exit 1
fi

# Получаем общее количество MP4 файлов
total_files=$(find "$target_dir" -type f -name "*.mp4" | wc -l)
echo "Найдено MP4 файлов: $total_files"

if [ "$total_files" -eq 0 ]; then
    echo "Нет MP4 файлов для обработки"
    exit 0
fi

total_duration=0
processed_files=0

echo -n "Обработка файлов: "

# Ищем все .mp4 файлы рекурсивно в указанной директории
while IFS= read -r -d '' file; do
    # Получаем длительность видео в секундах
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)
    
    # Проверяем, что duration содержит число
    if [[ $duration =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        # Суммируем длительность
        total_duration=$(echo "$total_duration + $duration" | bc)
        processed_files=$((processed_files + 1))
        echo -n "."
    else
        echo -e "\nНе удалось получить длительность файла: $file"
    fi
done < <(find "$target_dir" -type f -name "*.mp4" -print0)

echo -e "\n\nОбработано файлов: $processed_files из $total_files"

if [ "$processed_files" -eq 0 ]; then
    echo "Не удалось обработать ни одного файла"
    exit 1
fi

# Преобразуем общую длительность в читаемый формат (чч:мм:сс)
hours=$(echo "$total_duration / 3600" | bc)
remaining=$(echo "$total_duration % 3600" | bc)
minutes=$(echo "$remaining / 60" | bc)
seconds=$(echo "$remaining % 60" | bc)

printf "Общая продолжительность всех MP4 видео: %02d:%02d:%05.2f\n" $hours $minutes $seconds