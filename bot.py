import os
import telebot
from logic import process_image
from config import TOKEN
import time

bot = telebot.TeleBot(TOKEN)

welcome_msg = "Привет! Отправьте мне фото, и я замажу номерные знаки машин на нем и лица"
tries = 0
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_chat_action(message.chat.id, 'typing')
    time.sleep(2)
    bot.send_message(message.chat.id, welcome_msg)

@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    time.sleep(2)
    bot.reply_to(message, welcome_msg)

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    global tries
    if tries <= 4:
        bot.send_chat_action(message.chat.id, 'typing')
        time.sleep(2)
        try:
            # Получаем файл фотографии
            file_id = message.photo[-1].file_id 
            file_info = bot.get_file(file_id)
            file_ext = '.jpg'

            # Скачиваем файл
            downloaded_file = bot.download_file(file_info.file_path)

            # Сохраняем файл локально
            input_file_path = f"input{file_ext}"
            with open(input_file_path, 'wb') as new_file:
                new_file.write(downloaded_file)

            # Задаем имя выходного файла
            output_file_path = f"output{file_ext}"

            # Обрабатываем файл 
            process_image(input_file_path, output_file_path)

            # Отправляем обратно результат в виде фото
            with open(output_file_path, 'rb') as processed_photo:
                bot.send_photo(message.chat.id, processed_photo)

            # Удаляем временные файлы
            os.remove(input_file_path)
            os.remove(output_file_path)

            # Удаляем сообщение с фото из чата
            bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
            tries = tries + 1
        except Exception as e:
           
            bot.send_message(message.chat.id, f"Произошла ошибка: {str(e)}")
    else:
        bot.send_message(message.chat.id, "Произошла ошибка: у вас закончились бесплатные обработки фото")

if __name__ == "__main__":
    bot.polling(none_stop=True)
