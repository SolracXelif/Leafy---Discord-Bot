import asyncio
import io
import os
import discord
import numpy as np
import pyttsx3
import requests
from discord.ext import commands
from PIL import Image, ImageOps

try:
    from tensorflow import keras  # type: ignore[import-not-found]
except ImportError:
    import tensorflow.keras as keras  # type: ignore[import-not-found]

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

speaker = pyttsx3.init()
speaker.setProperty('rate', 150)
speaker.setProperty('volume', 1)


def _select_spanish_voice(engine):
    voices = engine.getProperty('voices')
    if not voices:
        return None

    for voice in voices:
        name_text = ' '.join(
            str(value).lower()
            for value in (
                getattr(voice, 'id', ''),
                getattr(voice, 'name', ''),
                getattr(voice, 'gender', ''),
            )
        )
        language_values = getattr(voice, 'languages', []) or []
        language_text = ' '.join(
            value.decode('utf-8', errors='ignore').lower() if isinstance(value, (bytes, bytearray)) else str(value).lower()
            for value in language_values
        )

        if (
            'spanish' in name_text
            or 'español' in name_text
            or 'espanol' in name_text
            or language_text.startswith('es')
            or 'es-' in language_text
            or 'es_' in language_text
        ):
            return voice.id

    return voices[0].id


selected_voice = _select_spanish_voice(speaker)
if selected_voice:
    speaker.setProperty('voice', selected_voice)


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


def hablar(texto):
    speaker.say(texto)
    speaker.runAndWait()


async def hablar_async(texto):
    await asyncio.to_thread(hablar, texto)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS)
    return np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

def obtener_clima(city: str) -> str:
    response = requests.get(f"https://wttr.in/{city}?format=3")
    if response.status_code == 200:
        return response.text
    return "No pude obtener el clima en este momento."
    

@bot.event
async def on_ready():
    print(f'Bot conectado como {bot.user}')


@bot.command()
async def clima(ctx, *, city: str):
    weather_info = obtener_clima(city)
    await ctx.send(weather_info)
    await hablar_async(weather_info)


@bot.command()
async def speak(ctx, *, texto):
    await hablar_async(texto)


@bot.command()
async def analizar(ctx):
    if not ctx.message.attachments:
        await ctx.send('Adjunta una imagen para analizar.')
        return
    if model is None:
        await ctx.send(f'No pude cargar el modelo. Error: {model_error}')
        return

    attachment = ctx.message.attachments[0]
    if not attachment.filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
        await ctx.send('Adjunta una imagen válida.')
        return

    image_bytes = await attachment.read()
    prediction = model.predict(preprocess_image(image_bytes), verbose=0)[0]

    clean_probability = float(prediction[0]) * 100
    contamination_probability = 100 - clean_probability if len(prediction) < 2 else float(prediction[1]) * 100
    conclusion = 'Probablemente está limpio.' if clean_probability >= contamination_probability else 'Probablemente está contaminado.'

    result = (
        'Resultado del análisis:\n'
        f'- Limpio: {clean_probability:.1f}%\n'
        f'- Contaminado: {contamination_probability:.1f}%\n'
        f'Conclusión: {conclusion}'
    )
    await ctx.send(result)
    await hablar_async(result)


@bot.command()
async def pagina(ctx):
    await ctx.send('Aquí tienes el enlace: http://127.0.0.1:5500/templates/index.html')

@bot.command()
async def ayuda(ctx):
    help_text = (
        'Comandos disponibles:\n'
        '- !clima <ciudad>: Obtiene el clima de la ciudad especificada.\n'
        '- !speak <texto>: El bot hablará el texto proporcionado.\n'
        '- !analizar: Analiza una imagen adjunta para determinar si está limpia o contaminada.\n'
        '- !pagina: Proporciona el enlace a la página web del proyecto.\n'
        '- !ayuda: Muestra esta ayuda.'
        '- !comandos: Muestra la lista de comandos.'
    )
    await ctx.send(help_text)
    
@bot.command()
async def comandos(ctx):
    comandos_text = (
        'Lista de comandos:\n'
        '- !clima <ciudad>\n'
        '- !speak <texto>\n'
        '- !analizar\n'
        '- !pagina\n'
        '- !ayuda\n'
        '- !comandos'
    )
    await ctx.send(comandos_text)
    

bot.run('INSERT_TOKEN_HERE')