from pytubefix import YouTube
import os

youtube_url = "https://www.youtube.com/watch?v=S63_y6zqlTI"

# Create output directory if it doesn't exist
os.makedirs('voice', exist_ok=True)

try:
    yt = YouTube(youtube_url)
    audio = yt.streams.filter(only_audio=True).first()
    
    if audio:
        audio.download(output_path='voice', filename="audio.mp4")
        print("Audio downloaded successfully!")
    else:
        print("No audio stream found")
        
except Exception as e:
    print(f"Error: {e}")