"""
Download todos os modelos de IA para ./models/
Pasta models_public no Google Drive — acesso público, sem autenticação.
"""
import os, sys

os.makedirs("models", exist_ok=True)

# (destino, file_id_drive) — pasta models_public (pública)
MODELS = [
    # ── Tabular — PKL ─────────────────────────────────────────────────────
    ("models/sepsis_xgboost.pkl",              "19VNoSkZb43Av7ujaZTYO8Z4T6uqwOKPZ"),  # v2 AUC 0.8645
    ("models/ecg_lgbm_ptbxl.pkl",             "1ZqFavjvVsclQDP2yCep_1qb6MRXV_Pzr"),
    ("models/ecg_label_encoder.pkl",          "1h5y3wfw8HxazI6T4SLwJwfqXP8As8YtW"),
    ("models/cardiac_xgboost_v2_combined.pkl","1BrbaUR-c0mJl8n3flJA-o7VE497Hengl"),
    ("models/preeclampsia_gbm.pkl",            "1nQg1Ew5yRGhikzV2Y41C1ldIhObUh90W"),
    ("models/mortality_gbm.pkl",              "1okGRsGv73iAVvcNIsdCSwns5EQyR-Vl5"),
    ("models/readmissao_gbm.pkl",             "1c6skYBu_Xj5NIsXGsPG9SHluFJZqmy77"),
    ("models/deterioracao_gbm.pkl",           "1A1MTI_Hxpr59HgoQWQXNE_NCMRKVJKZs"),
    ("models/vitaldb_ihi_v2.pkl",             "1WSaLZgdwa_WgpuHlH3UR9fG7KIZu9Ce8"),
    ("models/eeg_epilepsy_combined_gbm.pkl",  "1Wkf2UoDNGmykx56Fqs_bGyti4mtsCe-n"),
    ("models/circor_cardiac_gbm.pkl",         "1MGyQBzBQKWI6SaSZ14l7svo-69FLag14"),
    ("models/lung_sound_ensemble.pkl",        "1KCnKz6qWP4Javkt0xZz-x1k_CUJ_u8xu"),
    # ── Imagem — EfficientNet-B0 PTH ──────────────────────────────────────
    ("models/skin_efficientnet_b0_gpu.pth",   "1djuI95JgSJt7nJ71mxFJsSLEU_cTljUI"),
    ("models/skin_label_encoder_gpu.pkl",     "1VqRx9AXBtQ95DhqBqn5yFzzi0XTenmqN"),
    ("models/eyepacs_efficientnet_b0.pth",    "1Ja7DSuWfck-v397bAPAUHyTS9XlWhqzN"),
    ("models/eyepacs_label_encoder.pkl",      "1z7WRNb1-yXNNe7WElnBX-FtoYO87kFzx"),
    ("models/chest_xray_efficientnet_b0.pth", "16O2beLazd8pXAKb6hizqEUSllppnilxE"),
    ("models/chest_xray_classes.pkl",         "1DEUCftm2NeWZD05JZGjVR9w26u3JDHAS"),
    ("models/brain_tumor_efficientnet_b0.pth","1KpZgTrJmCCuEFPvMJ1DtJLDm4RPWL6Pd"),
    ("models/brain_tumor_classes.pkl",        "1TNgmavi2roxO-RL5jOH2nTWnZLcKOXQt"),
    ("models/fractura_efficientnet_b0.pth",   "1IINhfX72E3dNTegn-P_rcUDNHAFsUkGj"),
    ("models/fractura_classes.pkl",           "1PFWtVQIeWqMS6Pi8KzIBabqDz5Pz_155"),
    ("models/glaucoma_efficientnet_b0.pth",   "18-IrMNiiFn7YYWTsA1AkIwGk5DAOMuBa"),
    ("models/glaucoma_classes.pkl",           "1shW2ybIELwdunfCvNr7BuMEDGVGY0lOb"),
    ("models/mamografia_efficientnet_b0.pth", "15rJ9FTgElUIaBdwEghEtR6sPdLs9DJyY"),
    ("models/mamografia_classes.pkl",         "1aPss3yzP2KfDb3R3f1qaURshdJlTWdzF"),
    # ── NIH ChestX-ray14 — DenseNet-121 ───────────────────────────────────
    ("models/chestxray14_densenet121_v2.pth", "14Zc7kRQB07JEGe9A6wqDpl92xNKKBq84"),
    ("models/chestxray14_classes.pkl",        "1dxslOlPDw7RwZQ_16y_E2vQaCxrfPqBv"),
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
            raise FileNotFoundError("arquivo vazio após download")
    except Exception as e:
        print(f"[ERR]  {dest}: {e}")
        errors.append(dest)

total = len(MODELS)
ok = total - len(errors)
print(f"\n{'='*60}")
print(f"Modelos: {ok}/{total} baixados com sucesso.")
if errors:
    print(f"Falhas ({len(errors)}): {errors}")
    print("Endpoints afetados retornarão HTTP 503.")
print("="*60)
