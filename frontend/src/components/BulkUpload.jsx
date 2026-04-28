import React, { useState, useCallback, useRef } from "react";
import { useDropzone } from "react-dropzone";
import Papa from "papaparse";
import toast from "react-hot-toast";
import {
  Upload,
  FileText,
  X,
  Play,
  Download,
  BarChart3,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  ThumbsUp,
  ThumbsDown,
  Minus,
  Info,
} from "lucide-react";
import clsx from "clsx";
import { predictCSV } from "../api";

const MAX_ROWS      = 500;
const MAX_FILE_MB   = 5;
const PAGE_SIZE     = 10;

const SENTIMENT_CONFIG = {
  Positive: { icon: ThumbsUp,   badge: "badge-positive", dot: "bg-emerald-500" },
  Negative: { icon: ThumbsDown, badge: "badge-negative", dot: "bg-red-500"     },
  Neutral:  { icon: Minus,      badge: "badge-neutral",  dot: "bg-amber-400"   },
};

function DropZone({ onFile, disabled }) {
  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop: useCallback((accepted, rejected) => {
      if (rejected.length > 0) {
        toast.error("Only .csv files are accepted.");
        return;
      }
      if (accepted.length > 0) onFile(accepted[0]);
    }, [onFile]),
    accept:   { "text/csv": [".csv"] },
    maxFiles: 1,
    maxSize:  MAX_FILE_MB * 1024 * 1024,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={clsx(
        "relative border-2 border-dashed rounded-xl p-10 text-center",
        "cursor-pointer transition-all duration-200",
        isDragActive && !isDragReject && "border-indigo-400 bg-indigo-50",
        isDragReject  && "border-red-400 bg-red-50",
        !isDragActive && !isDragReject && "border-slate-200 bg-slate-50 hover:border-indigo-300 hover:bg-indigo-50/40",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-3">
        <div className={clsx(
          "w-14 h-14 rounded-2xl flex items-center justify-center",
          isDragActive ? "bg-indigo-100" : "bg-white border border-slate-200",
        )}>
          <Upload className={clsx("w-7 h-7", isDragActive ? "text-indigo-500" : "text-slate-400")} />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-700">
            {isDragActive ? "Drop your CSV here" : "Drag & drop a CSV file"}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            or <span className="text-indigo-500 font-medium">click to browse</span>
            {" "}· Max {MAX_FILE_MB}MB · Up to {MAX_ROWS} rows
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-slate-400 bg-white
                        border border-slate-100 rounded-lg px-3 py-1.5">
          <Info className="w-3 h-3" />
          CSV must have a <code className="mx-1 px-1 bg-slate-100 rounded text-slate-600 font-mono">review</code>
          column. Optional:
          <code className="mx-1 px-1 bg-slate-100 rounded text-slate-600 font-mono">product_id</code>
        </div>
      </div>
    </div>
  );
}

function FileInfoBar({ file, rowCount, onRemove }) {
  const sizeKB = (file.size / 1024).toFixed(1);
  return (
    <div className="flex items-center justify-between px-4 py-3
                    bg-indigo-50 border border-indigo-200 rounded-xl">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 bg-white border border-indigo-200 rounded-lg
                        flex items-center justify-center flex-shrink-0">
          <FileText className="w-4 h-4 text-indigo-500" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-700 truncate max-w-xs">
            {file.name}
          </p>
          <p className="text-xs text-slate-400">
            {sizeKB} KB · {rowCount} review{rowCount !== 1 ? "s" : ""} detected
          </p>
        </div>
      </div>
      <button
        onClick={onRemove}
        className="p-1.5 rounded-lg text-slate-400 hover:text-red-500
                   hover:bg-red-50 transition-all"
        title="Remove file"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

function PreviewTable({ rows }) {
  const preview = rows.slice(0, 5);
  const cols    = Object.keys(preview[0] || {});

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full text-xs">
        <thead className="bg-slate-50 border-b border-slate-200">
          <tr>
            <th className="px-3 py-2 text-left font-semibold text-slate-500">#</th>
            {cols.map(c => (
              <th key={c} className="px-3 py-2 text-left font-semibold text-slate-500">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {preview.map((row, i) => (
            <tr key={i} className="hover:bg-slate-50">
              <td className="px-3 py-2 text-slate-400 tabular-nums">{i + 1}</td>
              {cols.map(c => (
                <td key={c} className="px-3 py-2 text-slate-600 max-w-xs truncate">
                  {row[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 5 && (
        <div className="px-3 py-2 text-xs text-slate-400 bg-slate-50 border-t border-slate-200">
          Showing 5 of {rows.length} rows
        </div>
      )}
    </div>
  );
}

function ResultsTable({ results, onExport }) {
  const [page,         setPage]         = useState(0);
  const [filterSentiment, setFilter]    = useState("all");

  const filtered = filterSentiment === "all"
    ? results
    : results.filter(r => r.sentiment === filterSentiment);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageData   = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const handleFilter = (val) => {
    setFilter(val);
    setPage(0);
  };

  return (
    <div className="space-y-3">

      {/* Controls */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        {/* Filter buttons */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {["all", "Positive", "Negative", "Neutral"].map(opt => (
            <button
              key={opt}
              onClick={() => handleFilter(opt)}
              className={clsx(
                "px-3 py-1 text-xs font-medium rounded-full border transition-all",
                filterSentiment === opt
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600"
              )}
            >
              {opt === "all" ? `All (${results.length})` : (
                <>
                  {opt} ({results.filter(r => r.sentiment === opt).length})
                </>
              )}
            </button>
          ))}
        </div>

        {/* Export button */}
        <button
          onClick={onExport}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                     text-slate-600 bg-white border border-slate-200 rounded-lg
                     hover:border-indigo-300 hover:text-indigo-600 transition-all"
        >
          <Download className="w-3.5 h-3.5" />
          Export CSV
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-200">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500">#</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500">Review</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500">Sentiment</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500">Confidence</th>
              {results[0]?.product_id && (
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500">Product</th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {pageData.map((row, i) => {
              const cfg   = SENTIMENT_CONFIG[row.sentiment];
              const pct   = Math.round(row.confidence * 100);
              const absIdx = page * PAGE_SIZE + i + 1;
              return (
                <tr key={i} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3 text-slate-400 tabular-nums text-xs">
                    {absIdx}
                  </td>
                  <td className="px-4 py-3 text-slate-600 max-w-sm">
                    <p className="line-clamp-2 text-xs leading-relaxed">{row.review}</p>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cfg.badge}>{row.sentiment}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={clsx("h-full rounded-full", cfg.dot)}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-500 tabular-nums">{pct}%</span>
                    </div>
                  </td>
                  {results[0]?.product_id && (
                    <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                      {row.product_id || "—"}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>
            Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40
                         hover:border-indigo-300 disabled:cursor-not-allowed transition-all"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="px-2 tabular-nums">
              {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
              className="p-1.5 rounded-lg border border-slate-200 disabled:opacity-40
                         hover:border-indigo-300 disabled:cursor-not-allowed transition-all"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryBar({ summary, total, inferenceMs, modelVersion }) {
  const stats = [
    { label: "Total",    value: total,                         color: "text-slate-700" },
    { label: "Positive", value: summary.Positive || 0,         color: "text-emerald-600" },
    { label: "Negative", value: summary.Negative || 0,         color: "text-red-600"     },
    { label: "Neutral",  value: summary.Neutral  || 0,         color: "text-amber-600"   },
    { label: "Time",     value: `${inferenceMs}ms`,            color: "text-slate-500"   },
    { label: "Model",    value: `v${modelVersion}`,            color: "text-indigo-600"  },
  ];

  return (
    <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
      {stats.map(s => (
        <div key={s.label} className="card text-center py-3">
          <p className={clsx("text-lg font-bold tabular-nums", s.color)}>{s.value}</p>
          <p className="text-xs text-slate-400 mt-0.5">{s.label}</p>
        </div>
      ))}
    </div>
  );
}

export default function BulkUpload({ apiOnline, onResults }) {
  const [file,        setFile]        = useState(null);
  const [parsedRows,  setParsedRows]  = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [progress,    setProgress]    = useState(0);
  const [response,    setResponse]    = useState(null);
  const [error,       setError]       = useState(null);

  const handleFile = (f) => {
    setFile(f);
    setResponse(null);
    setError(null);

    Papa.parse(f, {
      header:       true,
      skipEmptyLines: true,
      complete: (result) => {
        const rows = result.data;
        if (!rows[0]?.review) {
          toast.error("CSV must have a 'review' column.");
          setFile(null);
          return;
        }
        if (rows.length > MAX_ROWS) {
          toast(`Showing first ${MAX_ROWS} rows`, { icon: "⚠️" });
        }
        setParsedRows(rows.slice(0, MAX_ROWS));
        toast.success(`Loaded ${Math.min(rows.length, MAX_ROWS)} reviews`);
      },
      error: () => {
        toast.error("Failed to parse CSV. Please check the file format.");
        setFile(null);
      },
    });
  };

  const handleRemoveFile = () => {
    setFile(null);
    setParsedRows([]);
    setResponse(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file || !apiOnline) return;
    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const data = await predictCSV(file, (evt) => {
        if (evt.total) {
          setProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      });

      setResponse(data);
      onResults(data);
      toast.success(
        `Analyzed ${data.total} reviews in ${data.inference_time_ms}ms!`
      );
    } catch (err) {
      setError(err.message || "Prediction failed. Please try again.");
      toast.error(err.message || "Upload failed.");
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleExport = () => {
    if (!response) return;
    const rows = response.results.map((r, i) => ({
      index:      i + 1,
      review:     r.review,
      sentiment:  r.sentiment,
      confidence: Math.round(r.confidence * 100) + "%",
      prob_positive: Math.round((r.probabilities.Positive || 0) * 100) + "%",
      prob_negative: Math.round((r.probabilities.Negative || 0) * 100) + "%",
      prob_neutral:  Math.round((r.probabilities.Neutral  || 0) * 100) + "%",
      product_id: r.product_id || "",
    }));

    const csv  = Papa.unparse(rows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `sentiment_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Results exported!");
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">

      {/* Heading */}
      <div>
        <h2 className="text-xl font-bold text-slate-900">Bulk CSV Upload</h2>
        <p className="text-sm text-slate-500 mt-1">
          Upload a CSV file to analyze up to {MAX_ROWS} product reviews at once.
        </p>
      </div>

      {/* Drop zone or file info */}
      {!file ? (
        <div className="card">
          <DropZone onFile={handleFile} disabled={!apiOnline || loading} />
        </div>
      ) : (
        <div className="card space-y-4">
          <FileInfoBar
            file={file}
            rowCount={parsedRows.length}
            onRemove={handleRemoveFile}
          />

          {/* Preview table */}
          {parsedRows.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Preview (first 5 rows)
              </p>
              <PreviewTable rows={parsedRows} />
            </div>
          )}

          {/* Progress bar (during upload) */}
          {loading && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-slate-500">
                <span>Uploading & analyzing…</span>
                <span className="tabular-nums">{progress}%</span>
              </div>
              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                  style={{ width: `${progress || 5}%` }}
                />
              </div>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="flex items-start gap-2 px-4 py-3 bg-red-50
                            border border-red-200 rounded-lg text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Submit button */}
          {!response && (
            <div className="flex justify-end">
              <button
                onClick={handleSubmit}
                disabled={loading || !apiOnline || parsedRows.length === 0}
                className={clsx(
                  "flex items-center gap-2 px-6 py-2.5 text-sm font-semibold",
                  "rounded-lg transition-all",
                  (!loading && apiOnline && parsedRows.length > 0)
                    ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm"
                    : "bg-slate-100 text-slate-400 cursor-not-allowed"
                )}
              >
                {loading ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing…</>
                ) : (
                  <><Play className="w-4 h-4" /> Analyze {parsedRows.length} Reviews</>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Results section */}
      {response && (
        <div className="space-y-5">

          {/* Success header */}
          <div className="flex items-center gap-2 text-emerald-700">
            <CheckCircle2 className="w-5 h-5" />
            <span className="font-semibold">Analysis Complete</span>
            <button
              onClick={() => { setFile(null); setParsedRows([]); setResponse(null); }}
              className="ml-auto flex items-center gap-1.5 text-xs text-slate-500
                         hover:text-indigo-600 px-3 py-1.5 border border-slate-200
                         rounded-lg hover:border-indigo-300 transition-all"
            >
              <Upload className="w-3 h-3" />
              Upload another
            </button>
          </div>

          {/* Summary stats */}
          <SummaryBar
            summary={response.summary}
            total={response.total}
            inferenceMs={response.inference_time_ms}
            modelVersion={response.model_version}
          />

          {/* View dashboard CTA */}
          <div className="flex items-center gap-3 px-4 py-3 bg-indigo-50
                          border border-indigo-200 rounded-xl text-sm">
            <BarChart3 className="w-5 h-5 text-indigo-500 flex-shrink-0" />
            <div>
              <span className="font-semibold text-indigo-700">
                Dashboard updated!
              </span>
              <span className="text-indigo-600 ml-1">
                Switch to the Analytics tab to see charts and insights.
              </span>
            </div>
          </div>

          {/* Results table */}
          <div className="card">
            <h3 className="text-sm font-semibold text-slate-700 mb-4">
              Prediction Results
            </h3>
            <ResultsTable
              results={response.results}
              onExport={handleExport}
            />
          </div>
        </div>
      )}

    </div>
  );
}
