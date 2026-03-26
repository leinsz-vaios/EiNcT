import argparse
import json
import os
import zipfile
from pathlib import Path


def read_docx_text(path: Path):
    text_parts = []
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            xml = zf.read('word/document.xml').decode('utf-8', errors='ignore')
        xml = xml.replace('</w:p>', '\n').replace('<w:tab/>', '\t')
        import re
        clean = re.sub(r'<[^>]+>', '', xml)
        text_parts.append(clean)
    except Exception:
        text_parts.append('')
    return '\n'.join(text_parts).strip()


def read_text_like(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.docx':
        return read_docx_text(path)
    if suffix in {'.txt', '.md', '.json', '.csv'}:
        return path.read_text(encoding='utf-8', errors='ignore')
    return ''


def parse_args():
    p = argparse.ArgumentParser(description='Ingest strategy docs and videos into JSON knowledge file')
    p.add_argument('--docs-dir', default='strategy_docs')
    p.add_argument('--out', default='strategy_knowledge.json')
    p.add_argument('--video', action='append', default=[])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs = []

    if docs_dir.exists():
        for fp in docs_dir.iterdir():
            if not fp.is_file():
                continue
            content = read_text_like(fp)
            docs.append({
                'name': fp.name,
                'path': str(fp),
                'content': content[:20000],
            })

    strategy_items = [
        {'id': 'liquidity_purge', 'summary': 'Look for liquidity grabs before reversal entries.'},
        {'id': 'turtle_soup', 'summary': 'False breakout / stop raid then quick reversal.'},
        {'id': 'smart_money_reversal', 'summary': 'Bias shift with structure break and displacement.'},
        {'id': 'silver_bullet', 'summary': 'Execution model with precise intraday setup windows.'},
    ]

    out = {
        'generated_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'video_sources': args.video,
        'document_sources': [d['path'] for d in docs],
        'strategy_items': strategy_items,
        'documents': docs,
    }

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)

    print(f'Wrote {args.out} with {len(args.video)} videos and {len(docs)} docs.')
