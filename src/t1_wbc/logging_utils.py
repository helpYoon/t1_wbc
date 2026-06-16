"""Lightweight per-tick CSV logger for WBC diagnostics."""
import csv


class TickLogger:
    def __init__(self, path, fields):
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.fields = fields

    def log(self, row):
        self.w.writerow({k: row.get(k, "") for k in self.fields})

    def close(self):
        self.f.close()
