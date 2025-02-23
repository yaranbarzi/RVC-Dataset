import os
import gradio as gr
import yt_dlp
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil

# Create necessary directories
os.makedirs('temp', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

def clean_temp_directory():
    """Clean temporary directory before processing"""
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    os.makedirs('temp')

def download_youtube_audio(url):
    """Download audio from YouTube URL with additional options to bypass restrictions"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join('temp', '%(title)s.%(ext)s'),
        'restrictfilenames': True,
        # Add these options to help bypass restrictions
        'nocheckcertificate': True,
        'no_warnings': True,
        'extractaudio': True,
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        # Add custom headers to mimic a web browser
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Return the path of downloaded file
        return [os.path.join('temp', f) for f in os.listdir('temp') if f.endswith('.wav')]
    except Exception as e:
        # Add more detailed error handling
        if 'HTTP Error 403' in str(e):
            raise Exception("خطای دسترسی به یوتیوب. لطفا از VPN استفاده کنید یا لینک را بررسی کنید.")
        elif 'Video unavailable' in str(e):
            raise Exception("ویدیو در دسترس نیست. لطفا لینک را بررسی کنید.")
        else:
            raise Exception(f"خطا در دانلود: {str(e)}")

def handle_uploaded_files(files):
    """Handle uploaded files by copying them to temp directory"""
    file_paths = []
    for file in files:
        if file is not None:
            temp_path = os.path.join('temp', os.path.basename(file.name))
            shutil.copy(file.name, temp_path)
            file_paths.append(temp_path)
    return file_paths

def separate_audio(url_input, file_input, model_choice):
    """Separate audio from either YouTube URL or uploaded files"""
    try:
        clean_temp_directory()
        input_files = []
        
        # Handle YouTube URL
        if url_input and url_input.strip():
            input_files.extend(download_youtube_audio(url_input))
        
        # Handle uploaded files
        if file_input and len(file_input) > 0:
            input_files.extend(handle_uploaded_files(file_input))
        
        if not input_files:
            return "لطفاً یک لینک یوتیوب یا فایل صوتی آپلود کنید."

        models = {
            'BS-Roformer-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
            'BS-Roformer-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
            'Mel-Roformer-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'
        }

        for file in input_files:
            os.system(f'audio-separator "{file}" --model_filename {models[model_choice]} --output_dir=output')

        return "جداسازی با موفقیت انجام شد!"
    except Exception as e:
        return f"خطا: {str(e)}"

def combine_and_clean(use_uploaded_files, uploaded_files=None):
    """Combine and clean audio files"""
    try:
        audio_files = []
        if not use_uploaded_files:
            output_files = [f for f in os.listdir('output') if 'Vocals' in f]
            audio_files = [os.path.join('output', f) for f in output_files]
        elif uploaded_files:
            audio_files = [f.name for f in uploaded_files if f is not None]

        if len(audio_files) < 2:
            return "حداقل دو فایل صوتی نیاز است!"

        combined_audio = None
        for file in audio_files:
            audio = AudioSegment.from_file(file)
            if combined_audio is None:
                combined_audio = audio
            else:
                combined_audio += audio

        chunks = split_on_silence(
            combined_audio,
            min_silence_len=1000,
            silence_thresh=-40,
            keep_silence=100
        )

        final_audio = chunks[0]
        for chunk in chunks[1:]:
            final_audio += chunk

        output_path = "output/combined_vocals.wav"
        final_audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"

def process_audio(echo_reduction=0.9, presence=0.1):
    """Process the final audio"""
    try:
        input_path = "output/combined_vocals.wav"
        if not os.path.exists(input_path):
            return "لطفا ابتدا از بخش ترکیب صداها استفاده کنید"

        audio, sr = librosa.load(input_path, sr=44100, mono=True)
        
        echo_reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=echo_reduction,
            stationary=False,
            n_fft=2048,
            win_length=2048,
            n_std_thresh_stationary=1.2
        )

        b1, a1 = butter(2, [200/22050, 8000/22050], btype='band')
        b2, a2 = butter(2, 4000/22050, btype='high')
        
        filtered = filtfilt(b1, a1, echo_reduced)
        high_freq = filtfilt(b2, a2, echo_reduced) * 0.2
        enhanced = filtered + (high_freq * presence)
        
        final_audio = librosa.util.normalize(enhanced) * 0.95
        
        output_path = "output/final_processed.wav"
        sf.write(output_path, final_audio, sr, 'PCM_24')
        
        return output_path
    except Exception as e:
        return f"خطا: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="پردازشگر حرفه‌ای صدا") as app:
    gr.Markdown("# 🎵 پردازشگر حرفه‌ای صدا")
    
    with gr.Tab("جداسازی صدا"):
        gr.Markdown("لینک یوتیوب یا آپلود مستقیم فایل‌ها")
        url_input = gr.Textbox(label="لینک ویدیو (اختیاری)")
        file_input = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="آپلود فایل‌ها (اختیاری)"
        )
        model_choice = gr.Dropdown(
            choices=["BS-Roformer-1297", "BS-Roformer-1296", "Mel-Roformer-1143"],
            label="انتخاب مدل",
            value="BS-Roformer-1297"
        )
        separate_button = gr.Button("شروع جداسازی")
        separate_output = gr.Textbox(label="نتیجه")
        separate_button.click(separate_audio, [url_input, file_input, model_choice], separate_output)
    
    with gr.Tab("ترکیب صداها"):
        use_uploaded = gr.Checkbox(label="استفاده از فایل‌های آپلودی", value=False)
        audio_files = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="انتخاب فایل‌های صوتی"
        )
        combine_button = gr.Button("ترکیب و حذف سکوت")
        combined_output = gr.Audio(label="خروجی", autoplay=True)
        
        def process_and_play(use_uploaded, files):
            result = combine_and_clean(use_uploaded, files)
            return result, gr.update(autoplay=True)
            
        combine_button.click(
            process_and_play,
            [use_uploaded, audio_files],
            [combined_output, combined_output]
        )
    
    with gr.Tab("پردازش نهایی"):
        with gr.Row():
            echo_slider = gr.Slider(minimum=0.7, maximum=0.95, value=0.9, label="میزان حذف اکو")
            presence_slider = gr.Slider(minimum=0.1, maximum=0.3, value=0.1, label="میزان حضور صدا")
        process_button = gr.Button("شروع پردازش")
        final_output = gr.Audio(label="خروجی نهایی", autoplay=True)
        
        def process_and_autoplay(echo, presence):
            result = process_audio(echo, presence)
            return result, gr.update(autoplay=True)
            
        process_button.click(
            process_and_autoplay,
            [echo_slider, presence_slider],
            [final_output, final_output]
        )

    gr.Markdown("""
    ### 🎥 ما را در یوتیوب دنبال کنید
    [![YouTube Channel](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/@aigolden)
    """)

if __name__ == "__main__":
    app.launch(share=True)
