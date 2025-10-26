# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="pv52sCLBNv9s"
# Cultura_from_Sheets_v5 — Sheets → expansão → perfis → Desejado → GAP → MD + PDF
# Requisitos: gspread, gspread_dataframe, pandas, numpy, matplotlib, tabulate

# ===========================
# Config
# ===========================
SPREADSHEET_INPUT_ID  = "COLOQUE_AQUI"   # planilha origem (aba Leads/Leads_Clean)
SPREADSHEET_OUTPUT_ID = "COLOQUE_AQUI"   # planilha destino: abas CULT_*
TAB_IN   = "Leads_Clean"                 # ou "Leads"

# caminhos locais
TAXONOMY_CSV = "data/essencias_barrett_cvf_denison_iso_v3.csv"
CAT_PATHS = [
    "data/essencias_barrett_cvf_denison_iso_v3.csv",   # ok usar a mesma p/ cores/chakra/camadas
    "data/essencias_88_enriquecido.json",
]

# controles
MIN_N   = 5
SMOOTH  = 1.0
DEFAULT_WEIGHTS = {
    ("preselection","positive"):    0.5,
    ("preselection","negative"):   -0.5,
    ("selection_final","positive"): 1.0,
    ("selection_final","negative"): -1.0,
}

# Config

APPLY_MIN_N_FILTER = False  # True => filtra HEATMAP/ATTRS por times com n_min_ok
DESIRED_FROM_FINAL_POS = True  # usa só selection_final positivos no “Desejado”
WRITE_DESIRED_TO_SHEET = True  # grava abas CULT_DESEJADO e CULT_GAP_*

# --- filtros e escrita extra ---
APPLY_MIN_N_FILTER = False          # True => filtra HEATMAP/ATTRS para times com n_min_ok
DESIRED_FROM_FINAL_POS = True       # usa só selection_final (+) para construir "Desejado"
WRITE_DESIRED_TO_SHEET = True       # grava abas CULT_DESEJADO e CULT_GAP_*
MODE_FILTER_ENABLED = False         # True => aplica filtro por mode
MODE_ALLOWED = {"online", "terapeuta"}  # ajuste conforme necessário

# exportações
EXPORT_MD  = True
MD_OUT     = "/mnt/data/report_cultura_v5.md"
EXPORT_PDF = True
PDF_OUT    = "/mnt/data/report_cultura_v5.pdf"

# %% id="ZLBNXaTqpzRR"
# ===========================
# Auth Google (Colab)
# ===========================
import gspread, pandas as pd, numpy as np, os, json, re, unicodedata
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.colab import auth as colab_auth
colab_auth.authenticate_user()
import google.auth
creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets",
                                       "https://www.googleapis.com/auth/drive"])
gc = gspread.authorize(creds)
assert SPREADSHEET_INPUT_ID  != "COLOQUE_AQUI",  "Defina SPREADSHEET_INPUT_ID"
assert SPREADSHEET_OUTPUT_ID != "COLOQUE_AQUI", "Defina SPREADSHEET_OUTPUT_ID"
ss_in  = gc.open_by_key(SPREADSHEET_INPUT_ID)
ss_out = gc.open_by_key(SPREADSHEET_OUTPUT_ID)

# %% id="SL4yH3H8tbAJ"
# ===========================
# Utils
# ===========================
from pathlib import Path
from collections import defaultdict

def _norm(s:str)->str:
    s = str(s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-\_]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_selection_both(sel_str):
    parts = [p.strip() for p in str(sel_str or "").split("|") if p.strip()]
    pos = [p[3:].strip() for p in parts if p.startswith("(+)")]
    neg = [p[3:].strip() for p in parts if p.startswith("(-)")]
    return pos, neg

def load_catalog(paths):
    for p in paths:
        fp = Path(p)
        if not fp.exists(): continue
        if fp.suffix.lower() == ".json":
            data = json.loads(fp.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("items", [])
            df = pd.DataFrame(items)
        else:
            df = pd.read_csv(fp)
        if "id" not in df.columns:
            if "essence_id" in df.columns: df["id"] = df["essence_id"]
            elif "name" in df.columns:     df["id"] = df["name"].map(_norm)
            else:                          df["id"] = df.iloc[:,0].astype(str)
        if "color" not in df.columns and "cor" in df.columns: df["color"] = df["cor"]
        keep = ["id","color","chakra","camada","arquetipo","dominio"]
        for k in keep:
            if k not in df.columns: df[k] = None
        df = df[keep].copy()
        df["id_norm"] = df["id"].astype(str).str.lower()
        return df[["id_norm","color","chakra","camada","arquetipo","dominio"]]
    return None

def infer_domain_from_chakra(ch):
    ch = str(ch or "").strip().lower()
    if ch in {"ch4","4"}: return "Rel"
    if ch in {"ch3","3","ch6","6"}: return "Prof"
    if ch in {"ch1","1","ch5","5"}: return "Ene"
    return "Prof"

def read_leads(spreadsheet, tab="Leads_Clean"):
    ws = spreadsheet.worksheet(tab)
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
    for c in ["stage","kind","mode","tenant_id","team"]:
        if c in df.columns: df[c] = df[c].astype(str).str.lower()
    if "stage" not in df.columns:
        if "kind" in df.columns:
            k = df["kind"].astype(str).str.lower()
            df["stage"] = k.where(k.isin(["preselection","selection_final"]), "")
        else:
            df["stage"] = ""
    for c in ["timestamp_local","timestamp"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    if "session_effective" not in df.columns:
        df["session_effective"] = df.get("event_id_linked", df.get("event_id"))
    return df

def expand_all(df):
    df2 = df[df["stage"].isin(["preselection","selection_final"])].copy()
    out = []
    for _, row in df2.iterrows():
        stage = row["stage"]
        pos, neg = parse_selection_both(row.get("selection"))
        for nm, valence in [(pos,"positive"),(neg,"negative")]:
            for name in nm:
                out.append({
                    "tenant_id": row.get("tenant_id") or "DEFAULT",
                    "team": row.get("team") or "DEFAULT",
                    "session_effective": row.get("session_effective"),
                    "stage": stage,
                    "valence": valence,
                    "essence_name": name
                })
    out = pd.DataFrame(out)
    if len(out)==0: return out
    out["essence_id"] = out["essence_name"].map(_norm)
    out["id_norm"] = out["essence_id"].astype(str).str.lower()
    return out

def write_tab(ss, name, df):
    try:
        ws = ss.worksheet(name); ws.clear()
    except Exception:
        rows = max(100, (len(df)+10) if df is not None else 100)
        ws = ss.add_worksheet(title=name, rows=str(rows), cols="50")
    if df is None or len(df)==0: df = pd.DataFrame([{"info":"sem dados"}])
    set_with_dataframe(ws, df.reset_index(drop=True), include_index=False)


# %% id="N3xmU_90tg6p"
# ===========================
# Taxonomia completa
# ===========================
def load_taxonomy_full(p=Path(TAXONOMY_CSV)):
    tx = pd.read_csv(p)
    tx.columns = [c.strip().lower() for c in tx.columns]
    def _norm_local(s):
        s = str(s or "").strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        s = re.sub(r"[^a-z0-9\s\-\_]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    tx["_key"] = tx["essencia"].map(_norm_local)
    for c in ["tema_primario","barrett_principal","capacidade_negocio","cvf_quadrante","denison_dimensao",
              "cor_primaria","polaridade_cor","camada","familia_botanica","limitante","barrett_adjacentes",
              "peso_aspiracional","chakra","arquetipos"]:
        if c not in tx.columns: tx[c] = None
    tx["peso_aspiracional"] = pd.to_numeric(tx.get("peso_aspiracional",1.0), errors="coerce").fillna(1.0)
    return tx

TAX = load_taxonomy_full()

# Mapear arquetipos → arquetipo se necessário
if "arquetipo" not in TAX.columns and "arquetipos" in TAX.columns:
    TAX["arquetipo"] = TAX["arquetipos"]

# Validação de cobertura do dicionário
def validate_taxonomy(expanded_df, tax_df):
    if "id_norm" not in expanded_df.columns:
        return pd.DataFrame({"missing_keys": []}), pd.DataFrame(columns=["id_norm","essencia"])
    keys = expanded_df["id_norm"].dropna().unique()
    tx_keys = set(tax_df["_key"].astype(str).str.lower()) if "_key" in tax_df.columns else set()
    missing = [k for k in keys if k not in tx_keys]
    # checar campos críticos
    need_cols = ["tema_primario","barrett_principal","capacidade_negocio","cor_primaria","polaridade_cor","camada","chakra","arquetipo"]
    coverage = []
    # juntar meta
    meta = expanded_df.merge(
        tax_df.rename(columns={"_key":"id_norm"})[
            ["id_norm","essencia"] + [c for c in need_cols if c in tax_df.columns]
        ],
        on="id_norm", how="left"
    )
    grp = meta.groupby("id_norm", dropna=False)
    for k, sub in grp:
        has_ess = "essencia" in sub and sub["essencia"].notna().any()
        ess_val = sub.loc[sub["essencia"].notna(), "essencia"].iloc[0] if has_ess else None
        row = {"id_norm": k, "essencia": ess_val}
        for c in need_cols:
            row[f"has_{c}"] = bool(sub[c].notna().any()) if c in sub.columns else False
        coverage.append(row)
    cov_df = pd.DataFrame(coverage)
    if cov_df.empty:
        cov_df = pd.DataFrame(columns=["id_norm","essencia"] + [f"has_{c}" for c in need_cols])
    return pd.DataFrame({"missing_keys": missing}), cov_df


# %% id="RoqHFWOGtk7Z"
# ===========================
# Scoring e agregação
# ===========================
def score_group(rows: pd.DataFrame, weights=None, catalog: pd.DataFrame=None, smooth=1.0):
    weights = weights or DEFAULT_WEIGHTS
    base = rows.copy()

    if catalog is not None:
        base = base.merge(catalog, on="id_norm", how="left")

    tx_min = TAX[["_key","tema_primario","barrett_principal","capacidade_negocio",
                  "cor_primaria","polaridade_cor","camada","familia_botanica",
                  "limitante","chakra","arquetipo"]].rename(columns={"_key":"id_norm"})
    base = base.merge(tx_min, on="id_norm", how="left")

    if "dominio" in base.columns:
        base["dominio"] = base["dominio"].fillna(base["chakra"].map(infer_domain_from_chakra))
    else:
        base["dominio"] = base["chakra"].map(infer_domain_from_chakra)

    base["w"] = base.apply(lambda r: weights.get((str(r["stage"]), str(r["valence"])), 0.0), axis=1)

    attr_list = [
        ("barrett_principal","Barrett"),
        ("tema_primario","Tema"),
        ("capacidade_negocio","Capacidade"),
        ("cor_primaria","Cor"),
        ("polaridade_cor","Polaridade"),
        ("camada","Camada"),
        ("chakra","Chakra"),
        ("arquetipo","Arquetipo"),
        ("dominio","Dominio"),
    ]

    out = {}
    for col, _label in attr_list:
        if col not in base.columns:
            continue
        g = base.groupby(col, dropna=False)
        rows_ = []
        for k, sub in g:
            k2  = "—" if pd.isna(k) or str(k).strip()=="" else str(k)
            pos = (sub["valence"]=="positive").sum()
            neg = (sub["valence"]=="negative").sum()
            sc  = sub["w"].sum()
            denom = pos + neg + 2*smooth
            rows_.append({
                "attr": col, "value": k2, "score": sc, "pos": pos, "neg": neg, "count": len(sub),
                "pos_rate": round((pos + smooth)/denom, 4),
                "neg_rate": round((neg + smooth)/denom, 4),
            })
        out[col] = pd.DataFrame(rows_).sort_values(["score","count"], ascending=[False,False]).reset_index(drop=True)

    pre_pos = set(base[(base["stage"]=="preselection") & (base["valence"]=="positive")]["id_norm"])
    fin_pos = set(base[(base["stage"]=="selection_final") & (base["valence"]=="positive")]["id_norm"])
    kept = len(pre_pos & fin_pos)
    ret = round(100*kept/max(1,len(pre_pos)),2) if len(pre_pos)>0 else None
    neg_share = round(100*(base["valence"]=="negative").mean(),2) if len(base) else 0.0

    from math import log2

    smooth_local = smooth

    def _ent(v):
        if not len(v):
            return None
        total = sum(v) + smooth_local*len(v)
        H = 0.0
        for c in v:
            p = (c + smooth_local)/total
            H += -p*(log2(p) if p>0 else 0.0)
        Hmax = log2(len(v)) if len(v)>1 else 1.0
        return round(H/Hmax, 4)

    finp = base[(base["stage"]=="selection_final") & (base["valence"]=="positive")]
    ent_dom   = _ent(finp["dominio"].value_counts().tolist())
    ent_chak  = _ent(finp["chakra"].value_counts().tolist())

    ent = {"pos": {"pct_limitantes":0.0,"pct_camada2_3":0.0},
           "neg": {"pct_limitantes":0.0,"pct_camada2_3":0.0}}
    if "limitante" in base.columns and "camada" in base.columns:
        for v in ["positive","negative"]:
            sub = base[base["valence"]==v]
            if len(sub):
                ent[v]["pct_limitantes"] = float((sub["limitante"].astype(str)=="1").mean())
                ent[v]["pct_camada2_3"] = float(sub["camada"].astype(str).str.contains("Camada2|Camada3", case=False, na=False).mean())

    return out, {"retencao_pos": ret, "neg_share": neg_share,
                 "entropy_dom_final": ent_dom, "entropy_chakra_final": ent_chak,
                 "entropia": ent}

def aggregate_company(expanded_rows, weights=None, catalog=None, smooth=1.0, min_n=5, apply_min_filter=False):
    weights = weights or DEFAULT_WEIGHTS
    full = expanded_rows.copy()

    ov = []
    for (tenant, team), sub in full.groupby(["tenant_id","team"], dropna=False):
        rec = {"tenant_id":tenant or "DEFAULT", "team":team or "DEFAULT"}
        for st,val in [("preselection","positive"),("preselection","negative"),
                       ("selection_final","positive"),("selection_final","negative")]:
            rec[f"{st}_{val}"] = ((sub["stage"]==st) & (sub["valence"]==val)).sum()
        rec["sessions"] = sub["session_effective"].nunique() if "session_effective" in sub.columns else None
        ov.append(rec)
    overview_cols = ["tenant_id","team","preselection_positive","preselection_negative",
                     "selection_final_positive","selection_final_negative","sessions"]
    OVERVIEW = pd.DataFrame(ov)
    if len(OVERVIEW):
        OVERVIEW = OVERVIEW.sort_values(["tenant_id","team"]).reset_index(drop=True)
    else:
        OVERVIEW = pd.DataFrame(columns=overview_cols)

    blocks, reten, vies, attrs_rows, ent_rows = [], [], [], [], []
    for (tenant, team), sub in full.groupby(["tenant_id","team"], dropna=False):
        subN = len(sub)
        sc_tables, metrics = score_group(sub, weights=weights, catalog=catalog, smooth=smooth)
        for attr, df_attr in sc_tables.items():
            df2 = df_attr.copy()
            df2.insert(0,"team", team or "DEFAULT")
            df2.insert(0,"tenant_id", tenant or "DEFAULT")
            blocks.append(df2)
            for _, r in df2.iterrows():
                attrs_rows.append({"tenant_id":tenant or "DEFAULT","team":team or "DEFAULT",
                                   "attr":r["attr"],"value":r["value"],
                                   "score":r.get("score"),"count":r.get("count"),
                                   "pos":r.get("pos"),"neg":r.get("neg"),
                                   "pos_rate":r.get("pos_rate"),"neg_rate":r.get("neg_rate")})
        reten.append({"tenant_id":tenant or "DEFAULT","team":team or "DEFAULT","retencao_pos":metrics["retencao_pos"]})
        neg_pct = round(100*(sub["valence"]=="negative").mean(),2) if subN else 0.0
        vies.append({"tenant_id":tenant or "DEFAULT","team":team or "DEFAULT",
                     "n_registros":subN,
                     "n_sessoes": sub["session_effective"].nunique() if "session_effective" in sub.columns else None,
                     "pct_negativos":neg_pct,
                     "n_min_ok": subN >= min_n})
        ent_rows.append({
            "tenant_id": tenant or "DEFAULT",
            "team": team or "DEFAULT",
            "entropy_dom_final": metrics.get("entropy_dom_final"),
            "entropy_chakra_final": metrics.get("entropy_chakra_final")
        })

    HEATMAP = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=["tenant_id","team","attr","value","score","pos","neg","count","pos_rate","neg_rate"])
    RETENCAO = pd.DataFrame(reten)
    if len(RETENCAO):
        RETENCAO = RETENCAO.sort_values(["tenant_id","team"]).reset_index(drop=True)
    else:
        RETENCAO = pd.DataFrame(columns=["tenant_id","team","retencao_pos"])
    VIESES   = pd.DataFrame(vies)
    if len(VIESES):
        VIESES = VIESES.sort_values(["tenant_id","team"]).reset_index(drop=True)
    else:
        VIESES = pd.DataFrame(columns=["tenant_id","team","n_registros","n_sessoes","pct_negativos","n_min_ok"])
    ATTRS    = pd.DataFrame(attrs_rows, columns=["tenant_id","team","attr","value","score","count","pos","neg","pos_rate","neg_rate"])
    if len(ATTRS):
        ATTRS = ATTRS.sort_values(["tenant_id","team","attr","score"], ascending=[True,True,True,False]).reset_index(drop=True)
    if apply_min_filter:
        ok = set(map(tuple, VIESES.query("n_min_ok == True")[["tenant_id","team"]].to_numpy()))
        if len(HEATMAP):
            HEATMAP = HEATMAP[[ (r.tenant_id, r.team) in ok for _, r in HEATMAP.iterrows() ]].reset_index(drop=True)
        if len(ATTRS):
            ATTRS   = ATTRS[[   (r.tenant_id, r.team) in ok for _, r in ATTRS.iterrows() ]].reset_index(drop=True)
    ENTROPIA = pd.DataFrame(ent_rows, columns=["tenant_id","team","entropy_dom_final","entropy_chakra_final"]).sort_values(["tenant_id","team"]).reset_index(drop=True)
    return {"OVERVIEW": OVERVIEW, "HEATMAP": HEATMAP, "RETENCAO": RETENCAO, "VIESES": VIESES, "ATTRS": ATTRS, "ENTROPIA": ENTROPIA}


# %% id="NnBFrk8QtqKj"
# ===========================
# Execução principal
# ===========================
df = read_leads(ss_in, TAB_IN)
df = df.dropna(subset=["selection"])
df = df[df["stage"].isin(["preselection","selection_final"])]
if MODE_FILTER_ENABLED and "mode" in df.columns:
    df = df[df["mode"].isin(MODE_ALLOWED)]

expanded = []
for _, row in df.iterrows():
    stage = row["stage"]
    pos, neg = parse_selection_both(row.get("selection"))
    for nm, valence in [(pos,"positive"),(neg,"negative")]:
        for name in nm:
            expanded.append({
                "tenant_id": row.get("tenant_id") or "DEFAULT",
                "team": row.get("team") or "DEFAULT",
                "session_effective": row.get("session_effective"),
                "stage": stage,
                "valence": valence,
                "essence_name": name
            })
expanded = pd.DataFrame(expanded)
if len(expanded):
    expanded["essence_id"] = expanded["essence_name"].map(_norm)
    expanded["id_norm"] = expanded["essence_id"].astype(str).str.lower()
else:
    expanded["essence_id"] = pd.Series(dtype=str)
    expanded["id_norm"] = pd.Series(dtype=str)
print("Registros expandidos:", len(expanded))

MISSING, TAX_COVER = validate_taxonomy(expanded, TAX)
# escreve abas de validação (opcional)
try:
    write_tab(ss_out, "CULT_TAX_MISSING", MISSING)
    write_tab(ss_out, "CULT_TAX_COVER", TAX_COVER)
except Exception as e:
    print("Aviso: não foi possível escrever abas de TAX (ok seguir).", e)

CAT = load_catalog(CAT_PATHS)
packs = aggregate_company(expanded, catalog=CAT, min_n=MIN_N, smooth=SMOOTH, apply_min_filter=APPLY_MIN_N_FILTER)

write_tab(ss_out, "CULT_OVERVIEW", packs["OVERVIEW"])
write_tab(ss_out, "CULT_HEATMAP",  packs["HEATMAP"])
write_tab(ss_out, "CULT_RETENCAO", packs["RETENCAO"])
write_tab(ss_out, "CULT_VIESES",   packs["VIESES"])
write_tab(ss_out, "CULT_ATTRS",    packs["ATTRS"])
write_tab(ss_out, "CULT_ENTROPIA", packs["ENTROPIA"])

# %% id="mb25RL9Httnh"
# ===========================
# Desejado + GAP
# ===========================

def build_desired_auto(expanded_rows, tax_df, k_total=12):
    freq = expanded_rows["id_norm"].value_counts()
    if freq.empty:
        return tax_df.head(0).copy()
    tax_idx = tax_df.set_index("_key")
    pool = tax_idx.reindex(freq.index)
    pool["freq"] = freq
    pool = pool.dropna(how="all").copy()
    # priorizar Camada1 e evitar Camada3:
    pool_c1 = pool[pool["camada"].astype(str).str.contains("Camada1", case=False, na=False)]
    pool_noc3 = pool[~pool["camada"].astype(str).str.contains("Camada3", case=False, na=False)]
    top_c1 = pool_c1.sort_values("freq", ascending=False).index.tolist()
    desejado = top_c1[:4]  # sementes
    for k in pool_noc3.sort_values("freq", ascending=False).index:
        if k in desejado:
            continue
        desejado.append(k)
        if len(desejado) >= k_total:
            break
    des = tax_df.set_index("_key").loc[desejado].reset_index()
    return des

def dist_from_keys(keys, col, tax_df):
    meta = tax_df.set_index("_key").loc[keys]
    vc = meta[col].value_counts()
    return (vc / vc.sum()).sort_values(ascending=False) if vc.sum()>0 else vc

def gap_table(cur_dist, des_dist, title):
    idx = sorted(set(cur_dist.index) | set(des_dist.index))
    df = pd.DataFrame({
        "Atual %":    [float(cur_dist.get(i,0))*100 for i in idx],
        "Desejado %": [float(des_dist.get(i,0))*100 for i in idx],
    }, index=idx)
    df["Gap (pp)"] = (df["Desejado %"] - df["Atual %"]).round(1)
    df.index.name = title
    return df.round(1)

expanded_enriched = expanded.merge(
    TAX[["_key","essencia","tema_primario","barrett_principal","capacidade_negocio",
         "cor_primaria","polaridade_cor","camada","familia_botanica","limitante","chakra"]]
    .rename(columns={"_key":"id_norm"}), on="id_norm", how="left"
)

cur_bar = expanded_enriched["barrett_principal"].value_counts()
cur_bar = (cur_bar/cur_bar.sum()).sort_values(ascending=False) if cur_bar.sum()>0 else cur_bar
cur_tema = expanded_enriched["tema_primario"].value_counts()
cur_tema = (cur_tema/cur_tema.sum()).sort_values(ascending=False) if cur_tema.sum()>0 else cur_tema
cur_cap = expanded_enriched["capacidade_negocio"].value_counts()
cur_cap = (cur_cap/cur_cap.sum()).sort_values(ascending=False) if cur_cap.sum()>0 else cur_cap

base_for_desired = expanded_enriched
if DESIRED_FROM_FINAL_POS:
    base_for_desired = base_for_desired[(base_for_desired["stage"]=="selection_final") & (base_for_desired["valence"]=="positive")]
DES = build_desired_auto(base_for_desired, TAX, k_total=12)
DES.to_csv("/mnt/data/desejado_auto.csv", index=False)

des_bar = dist_from_keys(DES["_key"], "barrett_principal", TAX)
des_tema = dist_from_keys(DES["_key"], "tema_primario", TAX)
des_cap  = dist_from_keys(DES["_key"], "capacidade_negocio", TAX)

GAP_BARRETT = gap_table(cur_bar, des_bar, "Nível Barrett")
GAP_TEMAS   = gap_table(cur_tema, des_tema, "Tema")
GAP_CAPS    = gap_table(cur_cap, des_cap, "Capacidade")

GAP_BARRETT.to_csv("/mnt/data/gap_barrett.csv")
GAP_TEMAS.to_csv("/mnt/data/gap_temas.csv")
GAP_CAPS.to_csv("/mnt/data/gap_capacidades.csv")

if WRITE_DESIRED_TO_SHEET:
    write_tab(ss_out, "CULT_DESEJADO", DES[["essencia","tema_primario","barrett_principal","camada"]])
    write_tab(ss_out, "CULT_GAP_BARRETT", GAP_BARRETT.reset_index())
    write_tab(ss_out, "CULT_GAP_TEMAS",   GAP_TEMAS.reset_index())
    write_tab(ss_out, "CULT_GAP_CAPS",    GAP_CAPS.reset_index())


# %% id="1jvG6gxrtx8Z"
# ===========================
# Export MD (executivo)
# ===========================
def md_tbl(df: pd.DataFrame):
    if df is None or len(df)==0: return "_sem dados_\n"
    return df.to_markdown(index=False) + "\n"

if EXPORT_MD:
    lines = []
    lines.append("# Cultura — Relatório Executivo v5\n\n")
    lines.append("## Vieses de Coleta\n")
    lines.append(md_tbl(packs["VIESES"]))
    lines.append("## Retenção pré → final (+)\n")
    lines.append(md_tbl(packs["RETENCAO"]))
    lines.append("## Maiores saldos por grupo\n")
    dfh = packs["HEATMAP"].copy()
    if len(dfh):
        dfh["saldo"] = dfh["pos"] - dfh["neg"]
        top = dfh.sort_values(["tenant_id","team","saldo","score","count"],
                              ascending=[True,True,False,False,False]) \
                 .groupby(["tenant_id","team"]).head(8)
        lines.append(md_tbl(top[["tenant_id","team","attr","value","saldo","score","count","pos","neg"]]))
    else:
        lines.append("_sem dados_\n")
    lines.append("## Destaques globais por atributo (score)\n")
    attrs = packs["ATTRS"]
    if len(attrs):
        tops = []
        for attr in ["cor_primaria","chakra","camada","arquetipo","dominio","tema_primario","barrett_principal","capacidade_negocio"]:
            col = "attr" if " " not in attr else attr
            sub = attrs[attrs["attr"].isin([attr, attr.replace(" ","_")])].sort_values(["score","count"], ascending=[False,False]).head(5)
            if len(sub): tops.append(sub.assign(attr_group=attr)[["attr_group","tenant_id","team","value","score","count"]])
        if tops: lines.append(md_tbl(pd.concat(tops, ignore_index=True)))
        else:    lines.append("_sem dados_\n")
    else:
        lines.append("_sem dados_\n")
    lines.append("## Entropia por time (FINAL +)\n")
    lines.append(md_tbl(packs["ENTROPIA"]))
    lines.append("## HEATMAP com proporções suavizadas\n")
    cols = ["tenant_id","team","attr","value","score","count","pos","neg","pos_rate","neg_rate"]
    hm = packs["HEATMAP"][cols] if set(cols).issubset(packs["HEATMAP"].columns) else packs["HEATMAP"]
    lines.append(md_tbl(hm))

    # Desejado + GAP
    lines.append("## Desejado automático (12)\n")
    lines.append(DES[["essencia","tema_primario","barrett_principal","camada"]].to_markdown(index=False) + "\n")
    lines.append("## GAP — Barrett\n")
    lines.append(GAP_BARRETT.to_markdown() + "\n")
    lines.append("## GAP — Temas\n")
    lines.append(GAP_TEMAS.to_markdown() + "\n")
    lines.append("## GAP — Capacidades\n")
    lines.append(GAP_CAPS.to_markdown() + "\n")

    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print("MD salvo em:", MD_OUT)

# %% id="hKZ01133t2MI"
# ===========================
# Export PDF (executivo)
# ===========================
if EXPORT_PDF:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def _bar(ax, series, title):
        ax.clear()
        if series is None or len(series)==0:
            ax.axis('off'); ax.set_title(title+" (sem dados)", pad=6); return
        ax.bar(list(map(str, series.index)), series.values)
        ax.set_title(title, pad=6)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.margins(x=0.02)

    with PdfPages(PDF_OUT) as pdf:
        # Página 1 — Sumário
        fig = plt.figure(figsize=(8.27,11.69))
        fig.suptitle("Cultura — Relatório Executivo v5", y=0.98)
        ax1 = fig.add_axes([0.08,0.76,0.84,0.18])
        ax2 = fig.add_axes([0.08,0.54,0.84,0.18])
        ax3 = fig.add_axes([0.08,0.32,0.84,0.18])
        ax4 = fig.add_axes([0.08,0.10,0.84,0.18])

        _bar(ax1, cur_bar, "Barrett — atual")
        _bar(ax2, cur_tema, "Temas — atual")
        _bar(ax3, cur_cap,  "Capacidades — atual")

        ent = packs["VIESES"].copy()
        txt = []
        if "RETENCAO" in packs and len(packs["RETENCAO"]):
            r = packs["RETENCAO"]
            txt.append(f"Retenção (+) mediana: {r['retencao_pos'].median():.1f}%" if r['retencao_pos'].notna().any() else "Retenção (+): s/dados")
        if len(packs["VIESES"]):
            v = packs["VIESES"]
            txt.append(f"Negativos médio: {v['pct_negativos'].mean():.1f}%")
            txt.append(f"Times com N≥{MIN_N}: {(v['n_min_ok']).mean()*100:.0f}%")
        ax4.axis('off')
        ax4.text(0.0, 0.9, "Resumo", fontsize=12, weight='bold')
        y = 0.85
        for line in txt:
            ax4.text(0.0, y, f"• {line}", fontsize=10); y -= 0.08

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Página 2 — Desejado e GAPs
        fig = plt.figure(figsize=(8.27,11.69))
        ax1 = fig.add_axes([0.08,0.76,0.84,0.18])
        ax2 = fig.add_axes([0.08,0.54,0.84,0.18])
        ax3 = fig.add_axes([0.08,0.32,0.84,0.18])
        ax4 = fig.add_axes([0.08,0.10,0.84,0.18])

        _bar(ax1, des_bar, "Barrett — desejado")
        _bar(ax2, des_tema, "Temas — desejado")
        _bar(ax3, des_cap,  "Capacidades — desejado")

        # mini tabela textual com top 6 do desejado
        ax4.axis('off')
        ax4.text(0.0, 0.95, "Desejado automático (Top 6)", fontsize=12, weight='bold')
        for i,(ess,tema,bar,cam) in enumerate(DES[["essencia","tema_primario","barrett_principal","camada"]].head(6).itertuples(index=False), start=1):
            ax4.text(0.0, 0.95 - 0.12*i, f"{i}. {ess} — {tema} | {bar} | {cam}", fontsize=9)

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Página 3 — GAPs em barras
        def _gap_to_series(df, top=8):
            s = df["Gap (pp)"].sort_values(ascending=False)
            return s.head(top)

        fig = plt.figure(figsize=(8.27,11.69))
        ax1 = fig.add_axes([0.08,0.68,0.84,0.26])
        ax2 = fig.add_axes([0.08,0.37,0.84,0.26])
        ax3 = fig.add_axes([0.08,0.06,0.84,0.26])

        _bar(ax1, _gap_to_series(GAP_BARRETT), "GAP — Barrett (pp) maiores")
        _bar(ax2, _gap_to_series(GAP_TEMAS),   "GAP — Temas (pp) maiores")
        _bar(ax3, _gap_to_series(GAP_CAPS),    "GAP — Capacidades (pp) maiores")

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    print("PDF salvo em:", PDF_OUT)

print("Concluído.")

# %% id="l09t8d__6hOb"
# === AI PROMPT — CULTURA (salva .txt e escreve na planilha) ===
from datetime import datetime

def _md(df):
    import pandas as pd
    if df is None or len(df)==0: return "_sem dados_\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return pd.DataFrame(df).to_markdown(index=False)+"\n"

def _safe_cols(df, cols):
    have = [c for c in cols if c in df.columns]
    return df[have] if have else df

TENANT_F = "ALL"
TEAM_F   = "ALL"
MODE_F   = "ALL"
PERIODO  = "não informado"

try:
    # Se você tiver filtros definidos no notebook, preencha aqui
    TENANT_F = str(TENANT_FILTER) if 'TENANT_FILTER' in globals() else TENANT_F
    TEAM_F   = str(TEAM_FILTER)   if 'TEAM_FILTER'   in globals() else TEAM_F
    MODE_F   = ",".join(MODE_ALLOWED) if 'MODE_ALLOWED' in globals() else MODE_F
except Exception:
    pass

# Tabelas base (packs gerado anteriormente)
VIESES    = packs.get("VIESES")
RETENCAO  = packs.get("RETENCAO")
OVERVIEW  = packs.get("OVERVIEW")
HEATMAP   = packs.get("HEATMAP")
ATTRS     = packs.get("ATTRS")
ENTROPIA  = packs.get("ENTROPIA")

# Desejado/GAP se existirem
DES_TAB   = globals().get("DES", None)
GAP_BAR   = globals().get("GAP_BARRETT", None)
GAP_TEM   = globals().get("GAP_TEMAS", None)
GAP_CAP   = globals().get("GAP_CAPS", None)

# Colunas “essenciais” para HEATMAP no prompt
hm_cols = ["tenant_id","team","attr","value","score","count","pos","neg","pos_rate","neg_rate"]
HEATMAP_VIEW = _safe_cols(HEATMAP, hm_cols)

prompt = []
prompt.append("# Prompt de Análise de Cultura — v1 (preenchido)\n")
prompt.append("Você é um analista sênior de cultura. Gere diagnóstico e plano de ação a partir das tabelas a seguir.\n")
prompt.append(f"Filtros: tenant_id={TENANT_F}, team={TEAM_F}, mode={MODE_F}, período={PERIODO}\n\n")

prompt.append("## CULT_VIESES\n");    prompt.append(_md(VIESES))
prompt.append("## CULT_RETENCAO\n");  prompt.append(_md(RETENCAO))
prompt.append("## CULT_OVERVIEW\n");  prompt.append(_md(OVERVIEW))
prompt.append("## CULT_HEATMAP\n");   prompt.append(_md(HEATMAP_VIEW))
prompt.append("## CULT_ATTRS\n");     prompt.append(_md(ATTRS))
if ENTROPIA is not None:
    prompt.append("## CULT_ENTROPIA\n"); prompt.append(_md(ENTROPIA))
if DES_TAB is not None:
    prompt.append("## CULT_DESEJADO\n"); prompt.append(_md(DES_TAB[["essencia","tema_primario","barrett_principal","camada"]]))
if GAP_BAR is not None:
    prompt.append("## CULT_GAP_BARRETT\n"); prompt.append(_md(GAP_BAR.reset_index()))
if GAP_TEM is not None:
    prompt.append("## CULT_GAP_TEMAS\n");   prompt.append(_md(GAP_TEM.reset_index()))
if GAP_CAP is not None:
    prompt.append("## CULT_GAP_CAPS\n");    prompt.append(_md(GAP_CAP.reset_index()))

prompt.append("\n## Instruções ao modelo\n")
prompt.append("1) Valide n_min_ok e pct_negativos; 2) Classifique retenção (<30, 30–60, ≥60); ")
prompt.append("3) Liste Top 3 por score em Barrett, Temas, Capacidades, Domínio, Chakra, Camada, indicando tração vs resistência via pos/neg_rate; ")
prompt.append("4) Leia ENTROPIA (diversidade vs monocultura); 5) Síntese por time; 6) 1–3 alavancas por time com métrica; ")
prompt.append("7) Conecte ações aos maiores GAPs; 8) Alerte vieses/limitações e proponha reavaliação em 4–6 semanas.\n")

PROMPT_TXT = "\n".join(prompt)
out_path = f"/mnt/data/prompt_cultura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(PROMPT_TXT)
print("Prompt CULTURA salvo em:", out_path)

# Grava na Sheets (aba CULT_PROMPT)
try:
    import pandas as pd
    write_tab(ss_out, "CULT_PROMPT", pd.DataFrame([{"prompt": PROMPT_TXT}]))
except Exception as e:
    print("Aviso: não foi possível escrever CULT_PROMPT:", e)

--------------------------------------
Hoje sai em .txt e na Sheets. No Colab só aparece a mensagem “salvo em…”.
Se quiser ver o texto completo na saída do Colab, adicione estas linhas ao fim de cada bloco.

Cultura

Logo após print("Prompt CULTURA salvo em:", out_path):

from IPython.display import display, Markdown
display(Markdown("### Pré-visualização do prompt (CULTURA)"))
display(Markdown(f"```text\n{PROMPT_TXT}\n```"))


# %% id="jaTgNf7FOjE5"
Patches mínimos para v5.1:

Patch 2 — TAX: mapear “arquetipos” → “arquetipo”

Após TAX = load_taxonomy_full():

if "arquetipo" not in TAX.columns and "arquetipos" in TAX.columns:
    TAX["arquetipo"] = TAX["arquetipos"]

Patch 3 — score_group: proporções suavizadas

Substitua o loop que monta rows_ por esta versão:

for k, sub in g:
    k2  = "—" if pd.isna(k) or str(k).strip()=="" else str(k)
    pos = (sub["valence"]=="positive").sum()
    neg = (sub["valence"]=="negative").sum()
    sc  = sub["w"].sum()
    denom = pos + neg + 2*SMOOTH
    rows_.append({
        "attr": col, "value": k2, "score": sc, "pos": pos, "neg": neg, "count": len(sub),
        "pos_rate": round((pos + SMOOTH)/denom, 4),
        "neg_rate": round((neg + SMOOTH)/denom, 4),
    })

Patch 4 — ENTROPIA real e saída

Dentro de score_group após neg_share, adicione:

from math import log2
def _ent(v):
    if not len(v): return None
    total = sum(v) + SMOOTH*len(v)
    H = 0.0
    for c in v:
        p = (c+SMOOTH)/total
        H += -p*(log2(p) if p>0 else 0.0)
    Hmax = log2(len(v)) if len(v)>1 else 1.0
    return round(H/Hmax, 4)

finp = base[(base["stage"]=="selection_final") & (base["valence"]=="positive")]
ent_dom   = _ent(finp["dominio"].value_counts().tolist())
ent_chak  = _ent(finp["chakra"].value_counts().tolist())


E no return inclua:

return out, {"retencao_pos": ret, "neg_share": neg_share,
             "entropy_dom_final": ent_dom, "entropy_chakra_final": ent_chak}

Patch 5 — aggregate_company: aplicar MIN_N e gerar CULT_ENTROPIA

Troque a assinatura para:

def aggregate_company(expanded_rows, weights=None, catalog=None, smooth=1.0, min_n=5, apply_min_filter=False):


Após montar VIESES, crie ENTROPIA:

ent_rows.append({"tenant_id": t, "team": m,
                 "entropy_dom_final": metrics["entropy_dom_final"],
                 "entropy_chakra_final": metrics["entropy_chakra_final"]})


Depois de construir HEATMAP/ATTRS, aplique filtro se apply_min_filter:

if apply_min_filter:
    ok = set(map(tuple, VIESES.query("n_min_ok == True")[["tenant_id","team"]].to_numpy()))
    if len(HEATMAP): HEATMAP = HEATMAP[[ (r.tenant_id, r.team) in ok for _,r in HEATMAP.iterrows() ]]
    if len(ATTRS):   ATTRS   = ATTRS[[   (r.tenant_id, r.team) in ok for _,r in ATTRS.iterrows() ]]
ENTROPIA = pd.DataFrame(ent_rows).sort_values(["tenant_id","team"]).reset_index(drop=True)
return {"OVERVIEW": OVERVIEW, "HEATMAP": HEATMAP, "RETENCAO": RETENCAO, "VIESES": VIESES, "ATTRS": ATTRS, "ENTROPIA": ENTROPIA}


No call:

packs = aggregate_company(expanded, catalog=CAT, min_n=MIN_N, smooth=SMOOTH, apply_min_filter=APPLY_MIN_N_FILTER)


E escreva a nova aba:

write_tab(ss_out, "CULT_ENTROPIA", packs["ENTROPIA"])

Patch 6 — “Desejado” focado em FINAL(+)

No bloco Desejado + GAP, troque:

base_for_desired = expanded_enriched
if DESIRED_FROM_FINAL_POS:
    base_for_desired = base_for_desired[(base_for_desired["stage"]=="selection_final") & (base_for_desired["valence"]=="positive")]
DES = build_desired_auto(base_for_desired, TAX, k_total=12)

Patch 7 — Remover sets frágeis

Dentro de build_desired_auto, elimine dependência de CAMADA1_SET/CAMADA3_SET.
Use TAX:

pool = tax_df.set_index("_key").loc[freq.index].copy()
pool["freq"] = freq.values
# priorizar Camada1 e evitar Camada3:
pool_c1 = pool[pool["camada"].astype(str).str.contains("Camada1", case=False, na=False)]
pool_noc3 = pool[~pool["camada"].astype(str).str.contains("Camada3", case=False, na=False)]
top_c1 = pool_c1.sort_values("freq", ascending=False).index.tolist()
desejado = top_c1[:4]  # sementes
for k in pool_noc3.sort_values("freq", ascending=False).index:
    if k in desejado: continue
    desejado.append(k)
    if len(desejado) >= k_total: break
des = tax_df.set_index("_key").loc[desejado].reset_index()
return des

Patch 8 — Escrever Desejado e GAP na planilha

Após gerar DES, GAP_*:

if WRITE_DESIRED_TO_SHEET:
    write_tab(ss_out, "CULT_DESEJADO", DES[["essencia","tema_primario","barrett_principal","camada"]])
    write_tab(ss_out, "CULT_GAP_BARRETT", GAP_BARRETT.reset_index())
    write_tab(ss_out, "CULT_GAP_TEMAS",   GAP_TEMAS.reset_index())
    write_tab(ss_out, "CULT_GAP_CAPS",    GAP_CAPS.reset_index())

Patch 9 — MD: entropia e taxas

No MD, após “Destaques globais…”:

lines.append("## Entropia por time (FINAL +)\n")
lines.append(md_tbl(packs["ENTROPIA"]))
lines.append("## HEATMAP com proporções suavizadas\n")
cols = ["tenant_id","team","attr","value","score","count","pos","neg","pos_rate","neg_rate"]
hm = packs["HEATMAP"][cols] if set(cols).issubset(packs["HEATMAP"].columns) else packs["HEATMAP"]
lines.append(md_tbl(hm))


Resultado: v5.1 mantém tudo que você já tem e adiciona N-mínimo efetivo, proporções estáveis, entropia e rastreabilidade do “Desejado+GAP” na própria Sheets.

# %% id="Vl9XAsWpp2F2"
Aplique no seu notebook v5 (ou v5.1). Use exatamente os blocos abaixo.


PATCH B — Validação de taxonomia

Inserir uma nova célula após carregar TAX = load_taxonomy_full().

# Mapear arquetipos → arquetipo se necessário
if "arquetipo" not in TAX.columns and "arquetipos" in TAX.columns:
    TAX["arquetipo"] = TAX["arquetipos"]

# Validação de cobertura do dicionário
def validate_taxonomy(expanded_df, tax_df):
    keys = expanded_df["id_norm"].dropna().unique()
    tx_keys = set(tax_df["_key"].astype(str).str.lower()) if "_key" in tax_df.columns else set()
    missing = [k for k in keys if k not in tx_keys]
    # checar campos críticos
    need_cols = ["tema_primario","barrett_principal","capacidade_negocio","cor_primaria","polaridade_cor","camada","chakra","arquetipo"]
    coverage = []
    # juntar meta
    meta = expanded_df.merge(
        tax_df.rename(columns={"_key":"id_norm"})[
            ["id_norm","essencia"] + [c for c in need_cols if c in tax_df.columns]
        ],
        on="id_norm", how="left"
    )
    grp = meta.groupby("id_norm", dropna=False)
    for k, sub in grp:
        has_ess = "essencia" in sub and sub["essencia"].notna().any()
        ess_val = sub.loc[sub["essencia"].notna(), "essencia"].iloc[0] if has_ess else None
        row = {"id_norm": k, "essencia": ess_val}
        for c in need_cols:
            row[f"has_{c}"] = bool(sub[c].notna().any()) if c in sub.columns else False
        coverage.append(row)
    cov_df = pd.DataFrame(coverage)
    return pd.DataFrame({"missing_keys": missing}), cov_df

MISSING, TAX_COVER = validate_taxonomy(expanded, TAX)
# escreve abas de validação (opcional)
try:
    write_tab(ss_out, "CULT_TAX_MISSING", MISSING)
    write_tab(ss_out, "CULT_TAX_COVER", TAX_COVER)
except Exception as e:
    print("Aviso: não foi possível escrever abas de TAX (ok seguir).", e)

PATCH C — Filtro por mode

No bloco onde você prepara df a partir da planilha, logo após:

df = df.dropna(subset=["selection"])
df = df[df["stage"].isin(["preselection","selection_final"])]


adicione:

if MODE_FILTER_ENABLED and "mode" in df.columns:
    df = df[df["mode"].isin(MODE_ALLOWED)]

PATCH D — Proporções suavizadas e ENTROPIA

Dentro de score_group substitua a criação de linhas do agrupamento por:

for k, sub in g:
    k2  = "—" if pd.isna(k) or str(k).strip()=="" else str(k)
    pos = (sub["valence"]=="positive").sum()
    neg = (sub["valence"]=="negative").sum()
    sc  = sub["w"].sum()
    denom = pos + neg + 2*SMOOTH
    rows_.append({
        "attr": col, "value": k2, "score": sc, "pos": pos, "neg": neg, "count": len(sub),
        "pos_rate": round((pos + SMOOTH)/denom, 4),
        "neg_rate": round((neg + SMOOTH)/denom, 4),
    })


E logo após calcular neg_share, adicione ENTROPIA:

from math import log2
def _ent(v):
    if not v: return None
    total = sum(v) + SMOOTH*len(v)
    H = 0.0
    for c in v:
        p = (c + SMOOTH)/total
        H += -p*(log2(p) if p>0 else 0.0)
    Hmax = log2(len(v)) if len(v)>1 else 1.0
    return round(H/Hmax, 4)

finp = base[(base["stage"]=="selection_final") & (base["valence"]=="positive")]
ent_dom  = _ent(finp["dominio"].value_counts().tolist())
ent_chk  = _ent(finp["chakra"].value_counts().tolist())


E inclua no return:

return out, {"retencao_pos": ret, "neg_share": neg_share,
             "entropy_dom_final": ent_dom, "entropy_chakra_final": ent_chk}

PATCH E — Aplicar MIN_N e nova aba ENTROPIA

Na função aggregate_company, mude a assinatura:

def aggregate_company(expanded_rows, weights=None, catalog=None, smooth=1.0, min_n=5, apply_min_filter=False):


Crie ent_rows e preencha:

ent_rows = []
# dentro do loop por (tenant,team):
ent_rows.append({
    "tenant_id": tenant or "DEFAULT",
    "team": team or "DEFAULT",
    "entropy_dom_final": metrics["entropy_dom_final"],
    "entropy_chakra_final": metrics["entropy_chakra_final"]
})


Após montar HEATMAP e ATTRS, aplique filtro opcional:

if apply_min_filter:
    ok = set(map(tuple, VIESES.query("n_min_ok==True")[["tenant_id","team"]].to_numpy()))
    if len(HEATMAP):
        HEATMAP = HEATMAP[[ (r.tenant_id, r.team) in ok for _, r in HEATMAP.iterrows() ]]
    if len(ATTRS):
        ATTRS   = ATTRS[[   (r.tenant_id, r.team) in ok for _, r in ATTRS.iterrows() ]]
ENTROPIA = pd.DataFrame(ent_rows).sort_values(["tenant_id","team"]).reset_index(drop=True)
return {"OVERVIEW": OVERVIEW, "HEATMAP": HEATMAP, "RETENCAO": RETENCAO, "VIESES": VIESES, "ATTRS": ATTRS, "ENTROPIA": ENTROPIA}


E, na escrita:

write_tab(ss_out, "CULT_ENTROPIA", packs["ENTROPIA"])

PATCH F — “Desejado” focado e escrito

Substitua a seleção da base para desejado:

base_for_desired = expanded_enriched
if DESIRED_FROM_FINAL_POS:
    base_for_desired = base_for_desired[(base_for_desired["stage"]=="selection_final") & (base_for_desired["valence"]=="positive")]
DES = build_desired_auto(base_for_desired, TAX, k_total=12)


E depois de gerar DES e os GAP_*, grave na Sheets se habilitado:

if WRITE_DESIRED_TO_SHEET:
    write_tab(ss_out, "CULT_DESEJADO", DES[["essencia","tema_primario","barrett_principal","camada"]])
    write_tab(ss_out, "CULT_GAP_BARRETT", GAP_BARRETT.reset_index())
    write_tab(ss_out, "CULT_GAP_TEMAS",   GAP_TEMAS.reset_index())
    write_tab(ss_out, "CULT_GAP_CAPS",    GAP_CAPS.reset_index())

PATCH G — MD: entropia e taxas

No bloco de export MD, após “Destaques globais…”, acrescente:

lines.append("## Entropia por time (FINAL +)\n")
lines.append(md_tbl(packs["ENTROPIA"]))
lines.append("## HEATMAP com proporções suavizadas\n")
cols = ["tenant_id","team","attr","value","score","count","pos","neg","pos_rate","neg_rate"]
hm = packs["HEATMAP"][cols] if set(cols).issubset(packs["HEATMAP"].columns) else packs["HEATMAP"]
lines.append(md_tbl(hm))


Esses patches mantêm a polaridade ±, integram Barrett e adicionam controles de viés e leitura executiva coerentes com o método.

# %% id="1P6VbFrnCWTg"

# %% id="sXd9QvJzCXHi"
