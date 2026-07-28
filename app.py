import os
import shutil
import gradio as gr
import librosa
import noisereduce as nr
import numpy as np

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)


def clean_directory(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass


def download_youtube_audio(url):
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join("temp", "%(title)s.%(ext)s"),
        "restrictfilenames": True,
        "nocheckcertificate": True,
        "no_warnings": True,
        "extractaudio": True,
        "geo_bypass": True,
        "geo_bypass_country": "US",
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return [
        os.path.join("temp", f)
        for f in os.listdir("temp")
        if f.endswith(".wav")
    ]


def handle_uploaded_files(files):
    file_paths = []
    for file in files:
        if file is not None:
            temp_path = os.path.join("temp", os.path.basename(file.name))
            shutil.copy(file.name, temp_path)
            file_paths.append(temp_path)
    return file_paths


def separate_audio(url_input, file_input, model_choice):
    try:
        clean_directory("temp")
        input_files = []

        if url_input and url_input.strip():
            input_files.extend(download_youtube_audio(url_input))

        if file_input and len(file_input) > 0:
            input_files.extend(handle_uploaded_files(file_input))

        if not input_files:
            return "❌ لطفاً یک لینک یوتیوب یا فایل صوتی آپلود کنید."

        models = {
            "BS-Roformer-1297": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "BS-Roformer-1296": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "Mel-Roformer-1143": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        }

        for file in input_files:
            os.system(
                f'audio-separator "{file}" --model_filename'
                f" {models[model_choice]} --output_dir=output"
            )

        return (
            "✅ جداسازی با موفقیت انجام شد! فایل‌ها در پوشه output ذخیره شدند."
        )
    except Exception as e:
        return f"❌ خطا: {str(e)}"


def combine_and_clean(use_uploaded_files, uploaded_files=None):
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    try:
        audio_files = []
        if not use_uploaded_files:
            # جستجو برای تمامی فایل‌های وکال تولید شده
            output_files = [
                f
                for f in os.listdir("output")
                if "Vocals" in f or "vocals" in f
            ]
            audio_files = [os.path.join("output", f) for f in output_files]
        elif uploaded_files:
            audio_files = [f.name for f in uploaded_files if f is not None]

        if not audio_files:
            return None, "❌ هیچ فایل صوتی برای ترکیب یافت نشد!"

        combined_audio = AudioSegment.empty()
        for file in audio_files:
            audio = AudioSegment.from_file(file)
            combined_audio += audio

        # حذف سکوت
        chunks = split_on_silence(
            combined_audio,
            min_silence_len=800,
            silence_thresh=-40,
            keep_silence=150,
        )

        if not chunks:
            final_audio = combined_audio
        else:
            final_audio = sum(chunks)

        output_path = "output/combined_vocals.wav"
        final_audio.export(output_path, format="wav")
        return output_path, "✅ ترکیب و حذف سکوت با موفقیت انجام شد."
    except Exception as e:
        return None, f"❌ خطا: {str(e)}"


def process_audio(echo_reduction=0.9, presence=0.1):
    import soundfile as sf
    from scipy.signal import butter, filtfilt

    try:
        input_path = "output/combined_vocals.wav"
        if not os.path.exists(input_path):
            return (
                None,
                "❌ فایل combined_vocals.wav یافت نشد. ابتدا مرحله ترکیب را"
                " اجرا کنید.",
            )

        audio, sr = librosa.load(input_path, sr=44100, mono=True)

        echo_reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=echo_reduction,
            stationary=False,
            n_fft=2048,
            win_length=2048,
            n_std_thresh_stationary=1.2,
        )

        b1, a1 = butter(2, [200 / 22050, 8000 / 22050], btype="band")
        b2, a2 = butter(2, 4000 / 22050, btype="high")

        filtered = filtfilt(b1, a1, echo_reduced)
        high_freq = filtfilt(b2, a2, echo_reduced) * 0.2
        enhanced = filtered + (high_freq * presence)

        final_audio = librosa.util.normalize(enhanced) * 0.95

        output_path = "output/final_processed.wav"
        sf.write(output_path, final_audio, sr, "PCM_24")

        return output_path, "✅ پردازش نهایی با موفقیت انجام شد."
    except Exception as e:
        return None, f"❌ خطا: {str(e)}"


# Gradio Interface
with gr.Blocks(title="پردازشگر حرفه‌ای صدا") as app:
    gr.Markdown("# 🎵 پردازشگر حرفه‌ای صدا")

    with gr.Tab("۱. جداسازی صدا"):
        gr.Markdown("لینک یوتیوب یا آپلود مستقیم فایل‌ها")
        url_input = gr.Textbox(label="لینک ویدیو (اختیاری)")
        file_input = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="آپلود فایل‌ها (اختیاری)",
        )
        model_choice = gr.Dropdown(
            choices=[
                "BS-Roformer-1297",
                "BS-Roformer-1296",
                "Mel-Roformer-1143",
            ],
            label="انتخاب مدل",
            value="BS-Roformer-1297",
        )
        separate_button = gr.Button("شروع جداسازی", variant="primary")
        separate_output = gr.Textbox(label="وضعیت")
        separate_button.click(
            separate_audio,
            [url_input, file_input, model_choice],
            separate_output,
        )

    with gr.Tab("۲. ترکیب صداها"):
        use_uploaded = gr.Checkbox(
            label="استفاده از فایل‌های آپلودی مجزا (بجای خروجی تب قبل)",
            value=False,
        )
        audio_files = gr.File(
            file_count="multiple",
            file_types=["audio", ".mp3", ".wav", ".m4a", ".ogg", ".aac"],
            label="انتخاب فایل‌های صوتی",
        )
        combine_button = gr.Button("ترکیب و حذف سکوت", variant="primary")
        status_combine = gr.Textbox(label="وضعیت")
        combined_output = gr.Audio(label="خروجی صوتی")

        combine_button.click(
            combine_and_clean,
            inputs=[use_uploaded, audio_files],
            outputs=[combined_output, status_combine],
        )

    with gr.Tab("۳. پردازش نهایی"):
        with gr.Row():
            echo_slider = gr.Slider(
                minimum=0.7, maximum=0.95, value=0.9, label="میزان حذف اکو"
            )
            presence_slider = gr.Slider(
                minimum=0.1, maximum=0.3, value=0.1, label="میزان حضور صدا"
            )
        process_button = gr.Button("شروع پردازش", variant="primary")
        status_process = gr.Textbox(label="وضعیت")
        final_output = gr.Audio(label="خروجی نهایی")

        process_button.click(
            process_audio,
            inputs=[echo_slider, presence_slider],
            outputs=[final_output, status_process],
        )

if __name__ == "__main__":
    app.launch(share=True)
