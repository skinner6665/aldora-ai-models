"""
Download todos os modelos de IA para ./models/
Executar no build do Railway (via Dockerfile) ou localmente.
Requer: pip install gdown
"""
import os
import sys

os.makedirs("models", exist_ok=True)

# (destino, file_id_drive)
MODELS = [
    # ── Tabular — PKL ────────────────────────────────────────────────────────
    ("models/sepsis_xgboost.pkl",              "1QTSD2SAN1c_ak0Qdu2NEW6RBzEIfPXNq"),
    ("models/ecg_lgbm_ptbxl.pkl",             "1ZqFavjvVsclQDP2yCep_1qb6MRXV_Pzr"),
    ("models/cardiac_xgboost_v2_combined.pkl", "1BrbaUR-c0mJl8n3flJA-o7VE497Hengl"),
    ("models/preeclampsia_gbm.pkl",            "1nQg1Ew5yRGhikzV2Y41C1ldIhObUh90W"),
    ("models/mortality_gbm.pkl",               "1okGRsGv73iAVvcNIsdCSwns5EQyR-Vl5"),
    ("models/readmissao_gbm.pkl",              "1c6skYBu_Xj5NIsXGsPG9SHluFJZqmy77"),
    ("models/deterioracao_gbm.pkl",            "1A1MTI_Hxpr59HgoQWQXNE_NCMRKVJKZs"),
    ("models/vitaldb_ihi_v2.pkl",              "1WSaLZgdwa_WgpuHlH3UR9fG7KIZu9Ce8"),
    ("models/eeg_epilepsy_combined_gbm.pkl",   "1Wkf2UoDNGmykx56Fqs_bGyti4mtsCe-n"),
    ("models/circor_cardiac_gbm.pkl",          "1MGyQBzBQKWI6SaSZ14l7svo-69FLag14"),
    ("models/lung_sound_ensemble.pkl",         "1KCnKz6qWP4Javkt0xZz-x1k_CUJ_u8xu"),
    # ── ONNX — modelo + arquivo de dados externo ─────────────────────────────
    ("models/chest_xray.onnx",                "154M0mx75pL_0wgxyTkKIcucjw7Xlt4Gz"),
    ("models/chest_xray.onnx.data",           "17RDh89H_-uG4qd-ets2FJjhcPzw1f9zc"),
    ("models/skin_dermatologia.onnx",         "1EN0bDHA1dYl6S5k_ZhnX6GaPbwpXkKhz"),
    ("models/skin_dermatologia.onnx.data",    "1T30nuzkxRO1jPfEazncOUB7Zs1d8LsnU"),
    ("models/retinopatia.onnx",               "1CfTYRi7KMNCy_cm1bn92wu7IMLP-aTCq"),
    ("models/retinopatia.onnx.data",          "1s9T1VRDVuPECGV1EsumRNByAlObEqTwg"),
    ("models/hemorragia_cerebral.onnx",       "1edeoXwitxQTXImqgzEwv7SSO49nbYOio"),
    ("models/hemorragia_cerebral.onnx.data",  "1yCgZtVBaXfpGrjtdED_tZWLQVe0mfyEf"),
    ("models/fratura_ossea.onnx",             "1CGCppwD3XD-u8twSIFsUuJ4ZENs4t_A5"),
    ("models/fratura_ossea.onnx.data",        "1KCEW3MXmRhDrEcGsZdegzpsasXnhsd--"),
    ("models/glaucoma.onnx",                  "1HjWmoSAm-YrXPTj6X-2mTybisr3kwWTr"),
    ("models/glaucoma.onnx.data",             "1hc29_d6mndqpYl6pkZYIOGOETJt1xqc0"),
    ("models/mamografia.onnx",                "1Zqmt5Drnd2687DewDj1ko1hY52LKWbaN"),
    ("models/mamografia.onnx.data",           "1WFZO0e_MgykAlCux_MlSWd_-C4Lb6w3v"),
]

try:
    import gdown
except ImportError:
    print("[SETUP] Instalando gdown...")
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown

errors = []
for dest, file_id in MODELS:
    if os.path.exists(dest) and os.path.getsize(dest) > 100:
        print(f"[SKIP] {dest} ({os.path.getsize(dest) // 1024} KB)")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[DOWN] {dest} ...")
    try:
        gdown.download(url, dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 100:
            print(f"[OK]   {dest} ({os.path.getsize(dest) // 1024} KB)")
        else:
            raise FileNotFoundError("arquivo vazio ou não criado após download")
    except Exception as e:
        print(f"[ERR]  {dest}: {e}")
        errors.append(dest)

total = len(MODELS)
ok = total - len(errors)
print(f"\n{'=' * 60}")
print(f"Modelos: {ok}/{total} baixados com sucesso.")
if errors:
    print(f"Falhas ({len(errors)}): {errors}")
    print("Endpoints afetados retornarão HTTP 503 até que os arquivos sejam baixados.")
print("=" * 60)
