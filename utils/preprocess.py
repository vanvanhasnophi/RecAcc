
"""eleme -> criteo 风格转换脚本（并行、带进度与序列统计）

新增功能：
- 并行处理（进程池），通过 `--workers` 指定并发数（默认: 全部CPU）。
- 进度条展示（尝试使用 `tqdm`，不可用时回退为简单计数打印）。
- 序列字段处理：
  - 数值序列：输出统计量：len, uniq_count, sum, mean, std, min, max
  - 类别序列：输出序列长度、唯一类别数、以及 top-k 的 (token_hash, count) 对

输出格式：TSV（无表头）。对于序列会展开成多个列，顺序为：label, dense..., 对于非数组稀疏字段仍输出单一哈希列，对于数组字段按规则展开多列。
"""

import argparse
import csv
import hashlib
import math
import os
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple


def load_field_spec(path: str) -> List[Dict]:
	fields = []
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			# skip empty rows
			if not row:
				continue
			# skip header row if present (some field files include a header like 'cat_name,field_type,...')
			if i == 0 and row[0].strip().lower() in ('cat_name', 'name', 'field'):
				continue
			name = row[0].strip()
			field_type = row[1].strip() if len(row) > 1 else ''
			is_array = row[2].strip().lower() == 'true' if len(row) > 2 else False
			orig = row[3].strip() if len(row) > 3 else ''
			dist = row[4].strip() if len(row) > 4 else ''
			fields.append({'name': name, 'field_type': field_type, 'is_array': is_array, 'orig': orig, 'dist': dist})
	return fields


def stable_hash8(s: str) -> str:
	return hashlib.md5(s.encode('utf-8')).hexdigest()[:8]


def is_money_field(name: str) -> bool:
	low = name.lower()
	return any(k in low for k in ('price', 'amt', 'amount', 'total'))


def parse_numeric_list(parts: List[str]) -> List[float]:
	out = []
	for p in parts:
		if p == '' or p == '-1':
			continue
		try:
			out.append(float(p))
		except Exception:
			continue
	return out


def stats_numeric(seq: List[float]) -> Tuple[int, int, float, float, float, float, float]:
	# return len, uniq_count, sum, mean, std, min, max
	# All numeric outputs returned as integers (rounded)
	if not seq:
		return 0, 0, 0, 0, 0, 0, 0
	n = len(seq)
	uniq = len(set(seq))
	s_f = sum(seq)
	mean_f = s_f / n
	sumsq = sum(x * x for x in seq)
	var = max(0.0, sumsq / n - mean_f * mean_f)
	std_f = math.sqrt(var)
	mn_f = min(seq)
	mx_f = max(seq)
	# convert to ints by rounding
	s = int(round(s_f))
	mean = int(round(mean_f))
	std = int(round(std_f))
	mn = int(round(mn_f))
	mx = int(round(mx_f))
	return n, uniq, s, mean, std, mn, mx


def stats_categorical(parts: List[str], topk: int) -> Tuple[int, int, List[Tuple[str, int]]]:
	clean = [p for p in parts if p != '' and p != '-1']
	length = len(clean)
	uniq = len(set(clean))
	ctr = Counter(clean)
	top = ctr.most_common(topk)
	return length, uniq, top


def normalize_input_lines(lines: List[str]) -> List[str]:
	if not lines:
		return lines
	if lines[0].strip().startswith('```'):
		lines = lines[1:]
		if lines and lines[-1].strip().startswith('```'):
			lines = lines[:-1]
	return lines


def process_row(row: List[str], fields: List[Dict], schema_cols: List[Dict], cat_k: int, top_k: int):
	"""返回 (out_row_in_schema_order, mapping_entries)
	mapping_entries: list of (field_name, token, hash)
	"""
	mapping_entries = []
	# prepare dict for all output columns defaulting to empty string
	col_values = {col['col_name']: '' for col in schema_cols}

	# label handling (label assumed at field[0] if present)
	label = '0'
	if fields and fields[0]['name'] == 'label':
		label = row[0].strip() if len(row) > 0 else '0'
	col_values['label'] = label

	# fill dense fields
	for spec in fields:
		name = spec['name']
		if name == 'label':
			continue
		ftype = spec['field_type']
		is_arr = spec['is_array']
		# find the raw value by field index
		try:
			idx = fields.index(spec)
			val = row[idx].strip() if idx < len(row) else ''
		except Exception:
			val = ''

		# decide by dist_data_type if present
		dist = spec.get('dist', '').lower()
		is_numeric_field = bool(dist and 'int' in dist) or ftype == 'dense'
		is_hash_field = bool(dist and 'hash' in dist)

		if is_numeric_field and not is_arr:
			# numeric single value
			if val == '' or val == '-1':
				col_values[name] = '0'
				continue
			try:
				if '.' in val:
					if is_money_field(name):
						col_values[name] = str(int(round(float(val) * 100)))
					else:
						col_values[name] = str(int(round(float(val))))
				else:
					col_values[name] = str(int(val))
			except Exception:
				col_values[name] = '0'
		elif is_hash_field and not is_arr:
			# single value hashed
			if val == '' or val == '-1':
				col_values[f'{name}_hash'] = ''
			else:
				h = stable_hash8(val)
				col_values[f'{name}_hash'] = h
				mapping_entries.append((name, val, h))

	# handle sparse / array fields
	for spec in fields:
		name = spec['name']
		if name == 'label' or spec['field_type'] == 'dense':
			continue
		is_arr = spec['is_array']
		try:
			idx = fields.index(spec)
			val = row[idx].strip() if idx < len(row) else ''
		except Exception:
			val = ''

		if not is_arr:
			if val == '' or val == '-1':
				col_values[f'{name}_hash'] = ''
			else:
				h = stable_hash8(val)
				col_values[f'{name}_hash'] = h
				mapping_entries.append((name, val, h))
		else:
			parts = [p for p in val.split(';') if p != '']
			orig = spec.get('orig', '').lower()
			if 'int' in orig or 'float' in orig:
				# numeric sequence
				is_money = is_money_field(name)
				nums_f = parse_numeric_list(parts[:cat_k])
				if not nums_f:
					# len and uniq are counts (0), others left empty
					col_values[f'{name}_len'] = '0'
					col_values[f'{name}_uniq'] = '0'
					col_values[f'{name}_sum'] = ''
					col_values[f'{name}_mean'] = ''
					col_values[f'{name}_std'] = ''
					col_values[f'{name}_min'] = ''
					col_values[f'{name}_max'] = ''
				else:
					if is_money:
						nums = [int(round(x * 100)) for x in nums_f]
					else:
						nums = [int(round(x)) for x in nums_f]
					n, uniq, s, mean, std, mn, mx = stats_numeric(nums)
					col_values[f'{name}_len'] = str(n)
					col_values[f'{name}_uniq'] = str(uniq)
					col_values[f'{name}_sum'] = str(s)
					col_values[f'{name}_mean'] = str(mean)
					col_values[f'{name}_std'] = str(std)
					col_values[f'{name}_min'] = str(mn)
					col_values[f'{name}_max'] = str(mx)
			else:
				# categorical sequence
				length, uniq, top = stats_categorical(parts[:cat_k], top_k)
				col_values[f'{name}_len'] = str(length)
				col_values[f'{name}_uniq'] = str(uniq)
				for i in range(top_k):
					token_col = f'{name}_top{i+1}_token_hash'
					cnt_col = f'{name}_top{i+1}_count'
					if i < len(top):
						token, cnt = top[i]
						h = stable_hash8(token)
						col_values[token_col] = h
						col_values[cnt_col] = str(cnt)
						mapping_entries.append((name, token, h))
					else:
						col_values[token_col] = ''
						col_values[cnt_col] = '0'

	# produce ordered row according to schema_cols
	out_row = [col_values.get(col['col_name'], '') for col in schema_cols]
	return out_row, mapping_entries


def chunked(iterable: List, size: int):
	for i in range(0, len(iterable), size):
		yield iterable[i:i+size]


def process_chunk(rows: List[List[str]], fields: List[Dict], schema_cols: List[Dict], cat_k: int, top_k: int):
	out = []
	mapping = {}
	for r in rows:
		out_row, entries = process_row(r, fields, schema_cols, cat_k, top_k)
		out.append(out_row)
		for fname, token, h in entries:
			mapping.setdefault(fname, {})
			# keep first seen mapping for token
			if token not in mapping[fname]:
				mapping[fname][token] = h
	return out, mapping


def simple_progress(total):
	# returns a simple counter function closure
	count = {'n': 0}
	def tick(n=1):
		count['n'] += n
		print(f"processed: {count['n']}/{total}", end='\r')
	return tick


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--fields', default='eleme_fields.csv')
	p.add_argument('--input', default='eleme_sample.csv')
	p.add_argument('--output', default='eleme_criteo.tsv')
	p.add_argument('--cat_k', type=int, default=50, help='序列截断长度')
	p.add_argument('--top_k', type=int, default=5, help='类别序列 top-k')
	p.add_argument('--chunk_size', type=int, default=5000, help='每个任务的行数')
	p.add_argument('--workers', type=int, default=0, help='并行 worker 数，0 为 CPU 核心数')
	p.add_argument('--output-dir', default=os.path.join('output', 'preproc'), help='输出目录（默认: ./output/preproc）')
	# --head-sample: if provided without a number, use 20; provide an integer to set row count; 0 disables
	p.add_argument('--head-sample', nargs='?', const=20, type=int, default=0,
				   help='同时输出一份带表头的前N行样本到 <output>.head.tsv；不带值时默认为20；0 禁用')
	args = p.parse_args()

	# ensure output directory exists
	args.output_dir = args.output_dir or os.path.join('output', 'preproc')
	os.makedirs(args.output_dir, exist_ok=True)
	# normalize output filename (keep basename) and build full output path
	output_name = os.path.basename(args.output)
	output_path = os.path.join(args.output_dir, output_name)

	fields = load_field_spec(args.fields)

	with open(args.input, 'r', encoding='utf-8') as f:
		raw_lines = f.read().splitlines()
	raw_lines = normalize_input_lines(raw_lines)

	reader = csv.reader(raw_lines)
	rows = [r for r in reader if r]

	# build and write semantic table (schema) for output columns
	def build_schema(fields: List[Dict], top_k: int, cat_k: int):
		# Build schema with ordering: label, then ALL numeric-type columns, then ALL non-numeric (hash) columns
		cols_label = [{'col_name': 'label', 'source': 'label', 'output_type': 'label', 'description': 'sample label'}]
		numeric_cols = []
		non_numeric_cols = []

		for spec in fields:
			name = spec['name']
			if name == 'label':
				continue
			ftype = spec['field_type']
			is_arr = spec['is_array']
			dist = spec.get('dist', '').lower()
			# if dist indicates int32 treat as numeric; if hash treat as non-numeric hash
			if dist and 'int' in dist:
				is_numeric_field = True
			else:
				is_numeric_field = False
			is_hash_field = bool(dist and 'hash' in dist)
			if ftype == 'dense' or is_numeric_field:
				# dense always numeric
				numeric_cols.append({'col_name': name, 'source': name, 'output_type': 'dense' if ftype == 'dense' else 'seq_numeric' if is_arr else 'dense', 'description': f'dense/numeric feature from {name}'})
			else:
				if not is_arr:
					# single non-array field: if dist says hash -> hash column, else treat as numeric-like
					if is_hash_field:
						non_numeric_cols.append({'col_name': name + '_hash', 'source': name, 'output_type': 'sparse_hash', 'description': f'stable hash of {name}'})
					else:
						# fallback: present as numeric column
						numeric_cols.append({'col_name': name, 'source': name, 'output_type': 'sparse_numeric' if not is_numeric_field else 'dense', 'description': f'numeric from {name}'})
				else:
					# array field
					if is_numeric_field:
						# numeric sequence stats -> numeric columns
						stats_names = ['len', 'uniq', 'sum', 'mean', 'std', 'min', 'max']
						for sname in stats_names:
							numeric_cols.append({'col_name': f'{name}_{sname}', 'source': name, 'output_type': 'seq_numeric', 'description': f'numeric seq {sname} for {name}'})
					else:
						# categorical sequence: len & uniq & counts numeric; token hashes non-numeric
						numeric_cols.append({'col_name': f'{name}_len', 'source': name, 'output_type': 'seq_categorical', 'description': f'sequence length for {name}'})
						numeric_cols.append({'col_name': f'{name}_uniq', 'source': name, 'output_type': 'seq_categorical', 'description': f'unique count for {name}'})
						for i in range(top_k):
							# token hash is non-numeric
							non_numeric_cols.append({'col_name': f'{name}_top{i+1}_token_hash', 'source': name, 'output_type': 'seq_categorical_topk', 'description': f'top{i+1} token hash for {name}'})
							# counts are numeric
							numeric_cols.append({'col_name': f'{name}_top{i+1}_count', 'source': name, 'output_type': 'seq_categorical_topk_count', 'description': f'top{i+1} token count for {name}'})

		# final ordering: label, numeric_cols, non_numeric_cols
		return cols_label + numeric_cols + non_numeric_cols

	schema = build_schema(fields, args.top_k, args.cat_k)
	schema_json_path = output_path + '.schema.json'
	schema_tsv_path = output_path + '.schema.tsv'
	# write schema as a top-level JSON object with key "columns"
	with open(schema_json_path, 'w', encoding='utf-8') as sj:
		json.dump({'columns': schema}, sj, ensure_ascii=False, indent=2)

	total_rows = len(rows)
	if total_rows == 0:
		print('no rows to process')
		return

	# prepare chunks
	chunks = list(chunked(rows, args.chunk_size))
	n_workers = args.workers if args.workers > 0 else os.cpu_count() or 1

	# try tqdm
	try:
		from tqdm import tqdm
		use_tqdm = True
	except Exception:
		use_tqdm = False

	results = {}
	if use_tqdm:
		pbar = tqdm(total=total_rows, desc='processing')
	else:
		tick = simple_progress(total_rows)

	# collect global mapping from chunks
	global_mapping = {}
	with ProcessPoolExecutor(max_workers=n_workers) as exe:
		futures = {exe.submit(process_chunk, ch, fields, schema, args.cat_k, args.top_k): i for i, ch in enumerate(chunks)}
		with open(output_path, 'w', encoding='utf-8', newline='') as outf:
			writer = csv.writer(outf, delimiter='\t')
			for fut in as_completed(futures):
				out_chunk, mapping_chunk = fut.result()
				for out_row in out_chunk:
					writer.writerow(out_row)
				# merge mapping_chunk into global_mapping
				for fname, mp in mapping_chunk.items():
					g = global_mapping.setdefault(fname, {})
					for token, h in mp.items():
						if token not in g:
							g[token] = h
				if use_tqdm:
					pbar.update(len(out_chunk))
				else:
					tick(len(out_chunk))
	if use_tqdm:
		pbar.close()
	else:
		print('\nprocessing complete')

	# write mapping files
	map_json = output_path + '.hashmap.json'
	with open(map_json, 'w', encoding='utf-8') as mj:
		json.dump(global_mapping, mj, ensure_ascii=False, indent=2)

	print(f'schema written: {schema_json_path}')
	print(f'hash mapping written: {map_json}')

	# optionally write head sample with header and first N rows (N from args.head_sample)
	if getattr(args, 'head_sample', 0) and int(getattr(args, 'head_sample', 0)) > 0:
		head_count = int(args.head_sample)
		head_path = output_path + '.head.tsv'
		with open(head_path, 'w', encoding='utf-8', newline='') as hf:
			# write header from schema column names
			header = '\t'.join([c['col_name'] for c in schema])
			hf.write(header + '\n')
			# write first N lines from main output
			with open(output_path, 'r', encoding='utf-8') as outf_read:
				for i, line in enumerate(outf_read):
					if i >= head_count:
						break
					hf.write(line)
		print(f'head sample written: {head_path} ({head_count} rows)')


if __name__ == '__main__':
	main()
