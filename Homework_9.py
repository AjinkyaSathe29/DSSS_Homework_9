import torch
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import pipeline

BOT_TOKEN = '7777655045:AAGEM3Jqm-C1jIO3xYZOtKyIrUmFQBC-huM'

def greet_user(update: Update, context: CallbackContext):
    update.message.reply_text("Hi there! I'm your virtual assistant. How can I assist you today?")

def initialize_model():
    model_identifier = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_pipeline = pipeline(
        "text-generation",
        model=model_identifier,
        torch_dtype=torch.float16 if compute_device.type == "cuda" else torch.float32,
        device=0 if compute_device.type == "cuda" else -1
    )
    return model_pipeline

def process_user_input(update: Update, context: CallbackContext):
    user_text = update.message.text
    print(f"Received message: {user_text}")
    
    model = initialize_model()
    ai_response = model(user_text, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    update.message.reply_text(f"{ai_response}")

def run_bot():
    bot_updater = Updater(BOT_TOKEN)
    command_dispatcher = bot_updater.dispatcher
    
    command_dispatcher.add_handler(CommandHandler("start", greet_user))
    command_dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, process_user_input))
    
    bot_updater.start_polling()
    bot_updater.idle()

if __name__ == '__main__':
    print("Device in use:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    run_bot()
