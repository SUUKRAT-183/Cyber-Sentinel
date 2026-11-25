"""
setup_2fa.py - run ONCE to create the dashboard login (password + authenticator app).

    python setup_2fa.py

It creates auth_config.json (keep it private) and 2fa_qr.png.
Scan the QR code with Google Authenticator, Microsoft Authenticator or Authy.
"""
import argparse
import getpass
from pathlib import Path

from engines import auth

parser = argparse.ArgumentParser()
parser.add_argument("--username")
parser.add_argument("--password")
args = parser.parse_args()

username = args.username or input("Choose a username: ").strip()
password = args.password or getpass.getpass("Choose a password (8+ characters): ")
if not args.password:
    if getpass.getpass("Repeat the password: ") != password:
        raise SystemExit("Passwords do not match. Run the script again.")

secret = auth.create_config(username, password)
uri = auth.provisioning_uri()

qr_path = Path(__file__).resolve().parent / "2fa_qr.png"
try:
    import qrcode
    qrcode.make(uri).save(qr_path)
    print(f"\nQR code saved to: {qr_path}")
    print("Open it and scan it with your authenticator app.")
except ImportError:
    print("\nqrcode is not installed, so no QR image was made.")

print("\nIf you cannot scan the QR code, add the account manually with this key:")
print(f"    {secret}")
print("\nDone. Start the dashboard with:  streamlit run app.py")
