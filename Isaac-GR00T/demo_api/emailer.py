"""
Email sender using Resend API.
Set RESEND_API_KEY env var.
"""
import os
import resend

TASK_DISPLAY = {
    "open_drawer": "Open a Drawer",
    "close_drawer": "Close a Drawer",
    "place_in_closed_drawer": "Place Object in Closed Drawer",
    "pick_coke_can": "Pick Up a Coke Can",
    "pick_object": "Pick Up an Object",
    "move_near": "Move Object Near Target",
}

FROM_EMAIL = os.environ.get("FROM_EMAIL", "GR00T Demo <demo@yourdomain.com>")


def send_result_email(to: str, task: str, success: bool, video_url: str):
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print(f"[email] RESEND_API_KEY not set, skipping email to {to}")
        return

    resend.api_key = api_key
    task_name = TASK_DISPLAY.get(task, task.replace("_", " ").title())
    result_emoji = "✅" if success else "❌"
    result_text = "Success" if success else "Failed"

    html = f"""
<div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
  <h2 style="color: #333;">Your GR00T Robot Demo is Ready! 🤖</h2>
  <p>The GR00T robot attempted to: <strong>{task_name}</strong></p>
  <p style="font-size: 1.4em;">Result: {result_emoji} <strong>{result_text}</strong></p>
  <p>
    <a href="{video_url}" style="
      display: inline-block;
      padding: 12px 24px;
      background: #76b900;
      color: white;
      text-decoration: none;
      border-radius: 6px;
      font-weight: bold;
    ">Watch the Video</a>
  </p>
  <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;">
  <p style="color: #999; font-size: 0.85em;">
    Powered by NVIDIA GR00T N1.6 — a vision-language-action robot model.
  </p>
</div>
"""

    resend.Emails.send({
        "from": FROM_EMAIL,
        "to": [to],
        "subject": f"Your robot demo is ready! {result_emoji}",
        "html": html,
    })
    print(f"[email] Sent result to {to}: {result_text}")
