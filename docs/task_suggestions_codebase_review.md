# Codebase-Review: konkrete Aufgabenvorschläge

Dieses Dokument sammelt vier kleine, klar abgrenzbare Tickets aus einem schnellen Review der aktuellen Repository-Version.

## 1) Aufgabe: Tippfehler korrigieren

**Problem**
- In `docs/metrics.md` steht „offizal ARC score“ statt „official ARC score“.

**Vorgeschlagene Änderung**
- Tippfehler in der Metrik-Beschreibung korrigieren.

**Akzeptanzkriterien**
- Der String „offizal“ kommt im Repository nicht mehr vor.
- Die Metrik-Zeile ist sprachlich korrekt und unverändert inhaltlich.

## 2) Aufgabe: Programmierfehler beheben (CPU-freundlicher Checkpoint-Load)

**Problem**
- `scripts/eval_only.py` lädt Checkpoints mit `map_location="cuda"` und initialisiert das Modell zwingend in einem CUDA-Context.
- Dadurch bricht das Skript auf CPU-only Maschinen schon beim Laden/Initialisieren hart ab, obwohl ein klarer Fehlerhinweis oder CPU-Fallback besser wäre.

**Vorgeschlagene Änderung**
- Geräteauswahl explizit über `--device {cuda,cpu}` einführen (Default weiterhin `cuda`).
- Vor Modell-Setup prüfen, ob CUDA verfügbar ist; sonst mit verständlicher Fehlermeldung abbrechen oder optional auf CPU fallen.
- `torch.load(..., map_location=device)` und Tensor-/Model-Moves konsistent auf das gewählte Gerät umstellen.

**Akzeptanzkriterien**
- Auf Systemen ohne CUDA scheitert das Skript nicht mit internem Stacktrace, sondern mit klarer, kontrollierter Meldung (oder läuft mit `--device cpu`).
- Bestehendes Verhalten auf CUDA-Systemen bleibt unverändert.

## 3) Aufgabe: Dokumentations-Unstimmigkeit korrigieren

**Problem**
- `README.md` verweist in „Results snapshots“ und „Key findings“ auf Pfade wie `reports/TRM-EXP-01/runs/.../eval_report.json`.
- Im Repository liegen jedoch nur kompakte Dateien (`reports/TRM-EXP-01/results.json`, `reports/TRM-EXP-02/results.json`, `reports/TRM-EXP-03/negative_result.json`).

**Vorgeschlagene Änderung**
- README-Pfade auf die tatsächlich versionierten Dateien anpassen.
- Falls die `runs/...`-Struktur nur lokal erzeugt wird: dies explizit als lokale Reproduktionsstruktur kennzeichnen.

**Akzeptanzkriterien**
- Jeder in README erwähnte Report-Pfad existiert im Repo.
- Leser können ohne Rückfragen die im README genannten Snapshot-Dateien öffnen.

## 4) Aufgabe: Testqualität verbessern

**Problem**
- Es gibt aktuell keine automatisierten Tests für zentrale Auswerte-Helfer (z. B. Ranking/Hashing/Metrikaggregation).

**Vorgeschlagene Änderung**
- Kleine `pytest`-Suite ergänzen, mindestens für:
  - `grid_hash` (stabiler Hash für identische Grids, Unterschiede bei geänderter Form/Inhalt),
  - `_rank_candidates` in `scripts/eval_voting.py` (deterministische Ordnung für `count_avgq` und `sumq_count`),
  - per-output vs per-task Metrikaggregation mit minimalen synthetischen Testdaten.

**Akzeptanzkriterien**
- Tests laufen lokal mit `pytest` durch.
- Mindestens ein Regressionstest deckt ein tie-break-Szenario im Ranking ab.
- CI-/lokaler Lauf dokumentiert den Testaufruf.
