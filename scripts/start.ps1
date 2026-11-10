if (!(Test-Path .env)) { Copy-Item .env.example .env }
python scripts\generate_demo_data.py
python -m uvicorn app.main:app --reload
