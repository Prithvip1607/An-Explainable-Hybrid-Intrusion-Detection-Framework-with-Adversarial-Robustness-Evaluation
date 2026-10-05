from pathlib import Path
import zipfile
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = ['Label', 'label']
META_SOURCE = '__source_file'
META_ORIGINAL_LABEL = '__original_label'


def ensure_dataset(raw_dir: str):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    zips = sorted(raw.glob('*.zip'))
    if zips:
        # Extract only when CSVs are not already present.
        existing = list(raw.rglob('*.csv'))
        if not existing:
            with zipfile.ZipFile(zips[0], 'r') as z:
                z.extractall(raw)
    csvs = sorted(raw.rglob('*.csv'))
    if not csvs:
        raise FileNotFoundError(
            'No CICIDS2017 CSV files found. Put MachineLearningCSV.zip in data/raw/ '
            'or extract the CSV files there.'
        )
    return csvs


def find_label(columns):
    for c in LABEL_CANDIDATES:
        if c in columns:
            return c
    for c in columns:
        if str(c).strip().lower() == 'label':
            return c
    raise ValueError('Label column not found.')


def normalize_label(s):
    return s.astype(str).str.strip()


def sample_csvs(csvs, max_per_class=30000, seed=42):
    """Read CICIDS2017 CSVs in chunks and cap each ORIGINAL attack class.

    The original label is preserved so later analyses can report class
    distributions. Source-file metadata is retained for temporal/day-wise
    generalization experiments and is never used as a model feature.
    """
    rng = np.random.default_rng(seed)
    buckets = {}
    counts = {}
    for path in csvs:
        for chunk in pd.read_csv(path, low_memory=False, chunksize=100000):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            label = find_label(chunk.columns)
            chunk[label] = normalize_label(chunk[label])
            chunk[META_SOURCE] = path.name
            chunk[META_ORIGINAL_LABEL] = chunk[label]
            for cls, g in chunk.groupby(label, sort=False):
                used = counts.get(cls, 0)
                remaining = max_per_class - used
                if remaining <= 0:
                    continue
                if len(g) > remaining:
                    g = g.iloc[rng.choice(len(g), size=remaining, replace=False)]
                buckets.setdefault(cls, []).append(g)
                counts[cls] = used + len(g)
    if not buckets:
        raise ValueError('No labeled rows were found in the CICIDS2017 CSV files.')
    frames = [pd.concat(v, ignore_index=True) for v in buckets.values()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df



def sample_csvs_per_source(csvs, max_per_class=5000, seed=42):
    """Sample each source CSV independently while preserving both binary classes.

    This is used only for day-wise evaluation. The main experiment intentionally
    uses the global class cap in ``sample_csvs``. A global cap can exhaust the
    ATTACK quota in early files, leaving later weekday files attack-free; that
    makes a whole-day holdout impossible. Per-source sampling avoids that artifact.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for path in csvs:
        parts = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=100000):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            label = find_label(chunk.columns)
            chunk[label] = normalize_label(chunk[label])
            chunk[META_SOURCE] = path.name
            chunk[META_ORIGINAL_LABEL] = chunk[label]
            for cls, g in chunk.groupby(label, sort=False):
                if len(g) > max_per_class:
                    g = g.iloc[rng.choice(len(g), size=max_per_class, replace=False)]
                parts.append(g)
        if parts:
            frames.append(pd.concat(parts, ignore_index=True))
    if not frames:
        raise ValueError('No labeled rows were found for per-source sampling.')
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def _clean_frame(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    label = find_label(df.columns)
    df[label] = normalize_label(df[label])
    if META_ORIGINAL_LABEL not in df.columns:
        df[META_ORIGINAL_LABEL] = df[label]
    if META_SOURCE not in df.columns:
        df[META_SOURCE] = 'unknown'
    y = (df[label].str.upper() != 'BENIGN').astype(int)
    X = df.drop(columns=[label, META_SOURCE, META_ORIGINAL_LABEL], errors='ignore')
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    meta = df.loc[valid, [META_SOURCE, META_ORIGINAL_LABEL]].copy()
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    keep = ~X.duplicated(keep='first')
    return X.loc[keep], y.loc[keep], meta.loc[keep]


def clean_and_prepare(df, seed=42, test_size=0.25):
    """Conventional stratified random split with scaler fitted on train only."""
    X, y, meta = _clean_frame(df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype('float32')
    X_test_s = scaler.transform(X_test).astype('float32')
    return (
        X_train_s, X_test_s, y_train.to_numpy().astype('int64'), y_test.to_numpy().astype('int64'),
        list(X.columns), scaler, meta_train.reset_index(drop=True), meta_test.reset_index(drop=True)
    )



def clean_with_validation(df, seed=42, test_size=0.25, validation_size=0.15):
    """Random train/validation/test split with scaler fit ONLY on train."""
    X, y, meta = _clean_frame(df)
    X_train_full, X_test, y_train_full, y_test, meta_train_full, meta_test = train_test_split(
        X, y, meta, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_train_full, y_train_full, meta_train_full,
        test_size=validation_size, stratify=y_train_full, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype('float32')
    X_val_s = scaler.transform(X_val).astype('float32')
    X_test_s = scaler.transform(X_test).astype('float32')
    return (
        X_train_s, X_val_s, X_test_s,
        y_train.to_numpy().astype('int64'), y_val.to_numpy().astype('int64'), y_test.to_numpy().astype('int64'),
        list(X.columns), scaler,
        meta_train.reset_index(drop=True), meta_val.reset_index(drop=True), meta_test.reset_index(drop=True)
    )

def _day_key(name):
    """Return a CICIDS weekday label when present in a source filename."""
    s = str(name).lower()
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        if day in s:
            return day.capitalize()
    return 'Unknown'


def _choose_daywise_sources(meta, y, test_source_pattern=None, test_day=None, min_class_count=50):
    """Select a whole capture day (all source CSVs for that weekday).

    CICIDS2017 is distributed as several CSV segments per weekday. Holding out
    one CSV can accidentally produce a test set containing only BENIGN or only
    ATTACK traffic. This helper therefore groups files by weekday and chooses
    a whole day that contains both binary classes, preferring Friday -> Monday.
    """
    sources = sorted(meta[META_SOURCE].unique(), key=_source_sort_key)
    if len(sources) < 2:
        raise ValueError('Day-wise evaluation needs at least two source CSV files.')

    if test_source_pattern:
        rx = re.compile(test_source_pattern, re.I)
        test_sources = [s for s in sources if rx.search(s)]
        if not test_sources:
            raise ValueError(f'No source file matched pattern {test_source_pattern!r}')
        selected_day = _day_key(test_sources[0])
    elif test_day:
        selected_day = str(test_day).capitalize()
        test_sources = [s for s in sources if _day_key(s) == selected_day]
        if not test_sources:
            raise ValueError(f'No source CSVs found for --daywise-test-day={test_day!r}')
    else:
        # Prefer a complete weekday with both classes and at least a modest
        # number of examples from each class. This avoids the old failure mode.
        weekday_order = ['Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
        candidates = []
        for day in weekday_order:
            day_sources = [s for s in sources if _day_key(s) == day]
            if not day_sources:
                continue
            mask = meta[META_SOURCE].isin(day_sources)
            counts = pd.Series(y.loc[mask]).value_counts().to_dict()
            n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
            if n0 >= min_class_count and n1 >= min_class_count:
                candidates.append((weekday_order.index(day), day, day_sources, n0, n1))
        if not candidates:
            # Relax the minimum count but still require both classes.
            for day in weekday_order:
                day_sources = [s for s in sources if _day_key(s) == day]
                if not day_sources:
                    continue
                mask = meta[META_SOURCE].isin(day_sources)
                counts = pd.Series(y.loc[mask]).value_counts().to_dict()
                if 0 in counts and 1 in counts:
                    candidates.append((weekday_order.index(day), day, day_sources, int(counts[0]), int(counts[1])))
        if not candidates:
            raise ValueError('No complete CICIDS weekday contains both BENIGN and ATTACK classes.')
        _, selected_day, test_sources, n0, n1 = candidates[0]

    return test_sources, selected_day


def daywise_split(df, test_source_pattern=None, test_day=None):
    """Prepare a source-file/day-wise split without fitting preprocessing on test data.

    By default all CSV segments belonging to the latest available CICIDS2017
    weekday with both BENIGN and ATTACK traffic are held out. This is more
    faithful to a day-wise generalization test than holding out one attack-only
    CSV segment. The selected day/files are recorded in the output metadata.
    """
    X, y, meta = _clean_frame(df)
    test_sources, _ = _choose_daywise_sources(meta, y, test_source_pattern, test_day)
    test_mask = meta[META_SOURCE].isin(test_sources).to_numpy()
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError('Day-wise split produced an empty train or test set.')
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    meta_train, meta_test = meta.loc[train_mask], meta.loc[test_mask]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError('Selected day-wise split does not contain both BENIGN and ATTACK classes.')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype('float32')
    X_test_s = scaler.transform(X_test).astype('float32')
    return (
        X_train_s, X_test_s, y_train.to_numpy().astype('int64'), y_test.to_numpy().astype('int64'),
        list(X.columns), scaler, meta_train.reset_index(drop=True), meta_test.reset_index(drop=True), test_sources
    )



def daywise_with_validation(df, test_source_pattern=None, test_day=None, seed=42, validation_size=0.15):
    """Day-wise train/test split, then validation sampled from training days only."""
    X, y, meta = _clean_frame(df)
    test_sources, selected_day = _choose_daywise_sources(meta, y, test_source_pattern, test_day)
    test_mask = meta[META_SOURCE].isin(test_sources).to_numpy()
    train_mask = ~test_mask
    X_full_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_full_train, y_test = y.loc[train_mask], y.loc[test_mask]
    meta_full_train, meta_test = meta.loc[train_mask], meta.loc[test_mask]
    if y_full_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError('Selected day-wise split does not contain both BENIGN and ATTACK classes.')
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_full_train, y_full_train, meta_full_train,
        test_size=validation_size, stratify=y_full_train, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype('float32')
    X_val_s = scaler.transform(X_val).astype('float32')
    X_test_s = scaler.transform(X_test).astype('float32')
    return (
        X_train_s, X_val_s, X_test_s,
        y_train.to_numpy().astype('int64'), y_val.to_numpy().astype('int64'), y_test.to_numpy().astype('int64'),
        list(X.columns), scaler,
        meta_train.reset_index(drop=True), meta_val.reset_index(drop=True), meta_test.reset_index(drop=True), test_sources
    )


def _source_sort_key(name):
    s = str(name).lower()
    # Prefer date-like tokens when present, otherwise natural lexical order.
    dates = re.findall(r'(20\d{2}[-_]\d{1,2}[-_]\d{1,2})', s)
    if dates:
        return dates[0].replace('_', '-')
    nums = [int(x) for x in re.findall(r'\d+', s)]
    return (s, nums)


def class_distribution(df):
    label = find_label(df.columns)
    return df[label].astype(str).str.strip().value_counts().sort_values(ascending=False)
