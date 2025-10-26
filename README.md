# Analytics Notebook + Codex + Jupytext

## Visão geral
Par `.ipynb` ↔ `.py` no GitHub.  
Codex edita o `.py` por PR.  
Colab executa o `.ipynb`.  
`jupytext --sync` mantém os dois alinhados.

## Estrutura do repositório
```
/SEU_REPO
│
├── data/
│   ├── essencias_88_enriquecido.json
│   ├── essencias_barrett_cvf_denison_iso_v3.csv
│   └── team_map.csv            # opcional
│
├── notebooks/
│   ├── Analytics.ipynb
│   ├── report_demo.ipynb
│   └── Cultura_from_Sheets.ipynb
│
├── Analytics.py                # pareado com o .ipynb via jupytext (py:percent)
├── requirements.txt
└── README.md


```

### requirements.txt
```
pandas
numpy
gspread
gspread_dataframe
python-dateutil
```

## 1) Parear o notebook (uma vez)
No Colab ou local:
```bash
pip install jupytext nbstripout
jupytext --set-formats ipynb,py:percent Analitycs.ipynb
jupytext --sync Analitycs.ipynb
nbstripout --install   # opcional: diffs de .ipynb mais limpos
git add Analitycs.ipynb Analitycs.py
git commit -m "pair notebook with jupytext"
git push
```

## 2) Fluxo Colab → GitHub
Editar no Colab e sincronizar antes de commitar:
```python
# no Colab (célula)
!pip -q install jupytext
!jupytext --sync /content/SEU_REPO/Analitycs.ipynb
# depois: commit/push (se estiver clonado no Colab) ou File → Save a copy in GitHub
```

## 3) Fluxo Codex (PR no .py) → Colab
Após merge do PR:
```python
# no Colab (célula)
!git -C /content/SEU_REPO pull
!pip -q install jupytext
!jupytext --sync /content/SEU_REPO/Analitycs.ipynb
```
Abra e rode o notebook.

## 4) Uso do Codex
CLI (exemplo):
```bash
npm i -g @openai/codex
codex login
# no diretório do repo
codex run "Editar Analitycs.py (jupytext py:percent):
- Aplicar patches nas Células 6, 6B, 6C, 7B, 7C
- Manter df_all e df_pagos_canon
- Adicionar célula de CHECKS
- Rodar testes locais se houver
- Abrir PR"
```

### Prompts úteis
- Refatore a Célula 6 para criar `session_effective` e `is_paid` como especificado. Não renomear variáveis. Gerar diff e testes.
- Criar Célula 6B com análises de pré-seleção e retenção. Adicionar writers `_write_sheet` e abas PRE_*.
- Ajustar funil para contar leads apenas em `selection_final` e dedupar pagamentos por `stripe_payment_intent_id`.

## 5) Célula 0 de setup no Colab
**Caso A — abriu o `.ipynb` direto do GitHub:**
```python
# Célula 0 — setup Jupytext no Colab (rodar 1x por sessão)
!pip -q install jupytext nbstripout gspread gspread_dataframe python-dateutil
!jupytext --set-formats ipynb,py:percent /content/Analitycs.ipynb
!jupytext --sync /content/Analitycs.ipynb
!nbstripout --install
```

**Caso B — clonando o repositório:**
```python
# Célula 0 — setup + repo
!pip -q install jupytext nbstripout gspread gspread_dataframe python-dateutil
%cd /content
!git clone https://github.com/SEU_USER/SEU_REPO.git
%cd /content/SEU_REPO
!jupytext --set-formats ipynb,py:percent Analitycs.ipynb
!jupytext --sync Analitycs.ipynb
!nbstripout --install
```

## 6) Regra de ouro da sincronização
- Editou no Colab (.ipynb) → **rodar** `!jupytext --sync` antes de commit/push.  
- Merge/PR no GitHub (.py pelo Codex) → `git pull` no Colab → **rodar** `!jupytext --sync`.

Evite editar `.ipynb` e `.py` ao mesmo tempo entre syncs.  
Se renomear o notebook, rode `--set-formats` novamente.

## 7) Célula de CHECKS rápidos (opcional)
```python
# Célula de CHECKS — rode após df_all e df_pagos_canon e, se criadas, PRE_*
import pandas as pd
import numpy as np

def _ok(x): return "OK" if x else "FAIL"
issues = []

print("=== CHECK 1 — Stages em df_all ===")
try:
    vc = df_all["stage"].value_counts(dropna=False).to_frame("linhas"); display(vc)
    need = {"preselection","selection_final","paid"}; have = set(vc.index.astype(str))
    print("Stages mínimos:", _ok(need.issubset(have)))
    if not need.issubset(have): issues.append("Faltam stages esperados em df_all")
except Exception as e:
    issues.append(f"Erro CHECK 1: {e}")
print()

print("=== CHECK 2 — Dedupe Stripe ===")
try:
    intents = df_pagos_canon.get("stripe_payment_intent_id", pd.Series(dtype=str)).astype(str)
    nonempty = intents.ne("").sum(); nunq = intents[intents.ne("")].nunique()
    print(f"intents não vazios: {nonempty} | intents únicos: {nunq} |", _ok(nonempty == nunq))
    if nonempty != nunq: issues.append("Pagamentos duplicados por intent em df_pagos_canon")
except Exception as e:
    issues.append(f"Erro CHECK 2: {e}")
print()

print("=== CHECK 3 — Funil básico ===")
try:
    leads_sf = df_all.loc[df_all["stage"].eq("selection_final"), "session_effective"].nunique()
    vendas = len(df_pagos_canon); receita = float(df_pagos_canon.get("price_num", pd.Series()).sum())
    conv = round(100 * vendas / max(leads_sf, 1), 2)
    print(f"Leads (selection_final únicos): {leads_sf}")
    print(f"Vendas (pagos únicos):         {vendas}")
    print(f"Receita total:                  {receita:,.2f}")
    print(f"Conversão %:                    {conv}%")
    print("Conversão <= 100%:", _ok(conv <= 100.0))
    if conv > 100: issues.append("Conversão > 100%")
except Exception as e:
    issues.append(f"Erro CHECK 3: {e}")
print()

print("=== CHECK 4 — Pré-seleção disponível ===")
try:
    pre_rows = int((df_all["stage"] == "preselection").sum())
    fin_rows = int((df_all["stage"] == "selection_final").sum())
    print(f"Linhas preselection: {pre_rows} | selection_final: {fin_rows}")
except Exception as e:
    issues.append(f"Erro CHECK 4: {e}")
print()

print("=== CHECK 5 — Tabelas PRE_* (se criadas) ===")
try:
    pre_stats_ok = 'PRE_PRE_STATS' in globals() and isinstance(PRE_PRE_STATS, pd.DataFrame)
    ret_ok      = 'PRE_RETENCAO'  in globals() and isinstance(PRE_RETENCAO,  pd.DataFrame)
    dist_ok     = 'PRE_DIST_SEL'  in globals() and isinstance(PRE_DIST_SEL,  pd.DataFrame)
    quase_ok    = 'PRE_QUASE'     in globals() and isinstance(PRE_QUASE,     pd.DataFrame)
    print(f"PRE_PRE_STATS: {_ok(pre_stats_ok)} | PRE_RETENCAO: {_ok(ret_ok)} | PRE_DIST_SEL: {_ok(dist_ok)} | PRE_QUASE: {_ok(quase_ok)}")
    if pre_stats_ok: display(PRE_PRE_STATS.head(5))
    if ret_ok:       display(PRE_RETENCAO.head(5))
    if dist_ok:      display(PRE_DIST_SEL.head(5))
    if quase_ok:     display(PRE_QUASE.head(5))
except Exception as e:
    issues.append(f"Erro CHECK 5: {e}")
print()

print("=== CHECK 6 — selection_count na final ===")
try:
    dist = (df_all.loc[df_all["stage"].eq("selection_final")]
                 .groupby("selection_count")["session_effective"].nunique()
                 .rename("pessoas").reset_index().sort_values("selection_count"))
    display(dist)
    vals = set(dist["selection_count"].astype(int).tolist())
    valido = vals.issubset({0,1,2,3,4,5,6,7})
    print("Valores de selection_count válidos:", _ok(valido))
    if not valido: issues.append(f"selection_count fora do esperado: {sorted(vals)}")
except Exception as e:
    issues.append(f"Erro CHECK 6: {e}")
print()

print("=== CHECK 7 — Tenants detectados ===")
try:
    tenants = sorted([t for t in df_all.get("tenant_id", pd.Series()).astype(str).unique() if t])
    print(f"Tenants: {tenants if tenants else '[nenhum]'}")
except Exception as e:
    issues.append(f"Erro CHECK 7: {e}")
print()

print("=== RESULTADO FINAL ===")
print("TUDO OK" if not issues else "ISSUES: " + "; ".join(issues))
```

## 8) Makefile opcional
```
sync:
	jupytext --sync Analitycs.ipynb

setup:
	pip install -r requirements.txt jupytext nbstripout
	nbstripout --install
```

### Uso
```bash
make setup
make sync
```

## Solução de problemas
- Notebook não refletiu o PR: `git pull` + `jupytext --sync`.  
- Conflitos: escolha o arquivo “fonte” mais recente e rode `--sync`.  
- Células sumiram no `.py`: confirme formato `py:percent` no `--set-formats`.  
- Execução falhou no Colab: `pip install -r requirements.txt` antes de rodar.
