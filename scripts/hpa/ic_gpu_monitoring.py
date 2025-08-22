import subprocess
import time
import smtplib
from email.mime.text import MIMEText


def get_free_gpu_memory(gpu_id=0):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader", "-i", str(gpu_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
    free_mem_mb = int(result.stdout.strip().split("\n")[0])
    return free_mem_mb / 1024  # Convert to GB


def send_email(subject, body, sender, password, recipient):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print("✅ Email sent!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def main():
    print("🚀 Starting GPU monitoring...")

    THRESHOLD_GB = 10  # notify when this much memory is free
    CHECK_INTERVAL = 120  # check every 2 minutes

    while True:
        for gpu_id in range(5):  # checks GPU 0 to 4
            try:
                free_mem = get_free_gpu_memory(gpu_id=gpu_id)

                print(f"GPU {gpu_id} Free Memory: {free_mem:.2f} GB")

                if free_mem >= THRESHOLD_GB:
                    send_email(
                        subject="🎉 GPU is Free!",
                        body=f"The GPU {gpu_id} has {free_mem:.2f} GB available.",
                        sender="jmesanto91@gmail.com",
                        password="wlro viva hbbq nvul",
                        recipient="jmesanto91@gmail.com"
                    )
                    return  # Exit after notifying
            except Exception as e:
                print(f"⚠️ Error checking GPU {gpu_id}: {e}")
                continue

        print("=" * 60)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
