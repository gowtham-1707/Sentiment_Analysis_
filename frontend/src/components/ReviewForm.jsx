import React, { useState, useRef } from "react";
import toast from "react-hot-toast";
import {
  Send,
  RotateCcw,
  Copy,
  CheckCheck,
  ThumbsUp,
  ThumbsDown,
  Minus,
  Loader2,
  Lightbulb,
  Clock,
  Tag,
} from "lucide-react";
import clsx from "clsx";
import { predictSingle } from "../api";

const MAX_CHARS = 5000;

const SENTIMENT_CONFIG = {
  Positive: {
    icon:       ThumbsUp,
    color:      "text-emerald-600",
    bg:         "bg-emerald-50",
    border:     "border-emerald-200",
    bar:        "bg-emerald-500",
    badge:      "badge-positive",
    emoji:      "😊",
  },
  Negative: {
    icon:       ThumbsDown,
    color:      "text-red-600",
    bg:         "bg-red-50",
    border:     "border-red-200",
    bar:        "bg-red-500",
    badge:      "badge-negative",
    emoji:      "😞",
  },
  Neutral: {
    icon:       Minus,
    color:      "text-amber-600",
    bg:         "bg-amber-50",
    border:     "border-amber-200",
    bar:        "bg-amber-400",
    badge:      "badge-neutral",
    emoji:      "😐",
  },
};

const SAMPLE_REVIEWS = [
  "This product exceeded all my expectations! The build quality is superb and it arrived well ahead of schedule. Couldn't be happier with my purchase!",
  "Absolute garbage. Broke after two days of normal use. Customer service was unhelpful and refused to offer a refund. Stay far away from this brand.",
  "It's okay for the price point. Does what it's supposed to do, nothing more nothing less. Packaging was decent and delivery was on time.",
  "Blown away by the performance. Battery lasts all day, the display is stunning, and setup took less than 5 minutes. Best tech purchase this year.",
  "Not worth the hype. The features advertised don't work as described and the instructions are completely useless. Very disappointing.",
];


function ProbabilityBar({ label, value, config }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-medium">
        <span className={clsx("flex items-center gap-1", config.color)}>
          <span>{config.emoji}</span>
          {label}
        </span>
        <span className="text-slate-600 tabular-nums">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-700", config.bar)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function ResultCard({ result, onReset }) {
  const [copied, setCopied] = useState(false);
  const sentiment = result.data.sentiment;
  const cfg       = SENTIMENT_CONFIG[sentiment];
  const Icon      = cfg.icon;
  const confidence = Math.round(result.data.confidence * 100);

  const handleCopy = async () => {
    const text = `Review: ${result.data.review}\nSentiment: ${sentiment}\nConfidence: ${confidence}%`;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success("Copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={clsx(
      "rounded-xl border-2 p-6 space-y-5 animate-in fade-in slide-in-from-bottom-2",
      cfg.bg, cfg.border
    )}>

      {/* Header row */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className={clsx(
            "w-12 h-12 rounded-xl flex items-center justify-center shadow-sm",
            "bg-white border", cfg.border
          )}>
            <Icon className={clsx("w-6 h-6", cfg.color)} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={clsx("text-2xl font-bold", cfg.color)}>
                {sentiment}
              </span>
              <span className={cfg.badge}>{confidence}% confident</span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Analyzed in {result.inference_time_ms}ms
              {result.model_version !== "unknown" && (
                <> · Model v{result.model_version}</>
              )}
            </p>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={handleCopy}
            className="p-2 rounded-lg bg-white border border-slate-200
                       text-slate-400 hover:text-slate-700 hover:border-slate-300
                       transition-all"
            title="Copy result"
          >
            {copied
              ? <CheckCheck className="w-4 h-4 text-emerald-500" />
              : <Copy className="w-4 h-4" />
            }
          </button>
          <button
            onClick={onReset}
            className="p-2 rounded-lg bg-white border border-slate-200
                       text-slate-400 hover:text-slate-700 hover:border-slate-300
                       transition-all"
            title="Analyze another review"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Confidence gauge */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-slate-500 font-medium">
          <span>Confidence</span>
          <span className="tabular-nums">{confidence}%</span>
        </div>
        <div className="h-3 bg-white rounded-full overflow-hidden border border-slate-200">
          <div
            className={clsx("h-full rounded-full transition-all duration-1000", cfg.bar)}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* Probability bars for all 3 classes */}
      <div className="space-y-3">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Class Probabilities
        </p>
        {Object.entries(result.data.probabilities).map(([label, prob]) => (
          <ProbabilityBar
            key={label}
            label={label}
            value={prob}
            config={SENTIMENT_CONFIG[label]}
          />
        ))}
      </div>

      {/* Review text preview */}
      <div className="pt-1 border-t border-white/60">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
          Analyzed Review
        </p>
        <p className="text-sm text-slate-600 line-clamp-3 italic">
          "{result.data.review}"
        </p>
      </div>

    </div>
  );
}

export default function ReviewForm({ apiOnline }) {
  const [review,      setReview]      = useState("");
  const [productId,   setProductId]   = useState("");
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const textareaRef                   = useRef(null);

  const charCount  = review.length;
  const isOverLimit = charCount > MAX_CHARS;
  const canSubmit  = review.trim().length >= 3 && !isOverLimit && !loading && apiOnline;
  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setResult(null);
    try {
      const data = await predictSingle(review.trim(), productId.trim() || null);
      setResult(data);
      toast.success(`Prediction: ${data.data.sentiment} (${Math.round(data.data.confidence * 100)}%)`);
    } catch (err) {
      toast.error(err.message || "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setReview("");
    setProductId("");
    setResult(null);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  const handleSample = (sample) => {
    setReview(sample);
    setResult(null);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">

      {/* Page heading */}
      <div>
        <h2 className="text-xl font-bold text-slate-900">Single Review Analysis</h2>
        <p className="text-sm text-slate-500 mt-1">
          Enter a product review to classify it as Positive, Negative, or Neutral.
        </p>
      </div>

      {/* Main card */}
      <div className="card space-y-4">

        {/* Product ID (optional) */}
        <div>
          <label className="block text-xs font-semibold text-slate-500
                            uppercase tracking-wider mb-1.5">
            <Tag className="w-3 h-3 inline mr-1" />
            Product ID
            <span className="ml-1 font-normal text-slate-400 normal-case tracking-normal">
              (optional)
            </span>
          </label>
          <input
            type="text"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            placeholder="e.g. B001234XYZ"
            maxLength={100}
            className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg
                       bg-slate-50 text-slate-700 placeholder-slate-400
                       focus:outline-none focus:ring-2 focus:ring-indigo-500
                       focus:border-transparent transition-all"
          />
        </div>

        {/* Review textarea */}
        <div>
          <label className="block text-xs font-semibold text-slate-500
                            uppercase tracking-wider mb-1.5">
            Review Text
            <span className="ml-1 font-normal text-slate-400 normal-case tracking-normal">
              (3–5000 characters)
            </span>
          </label>
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={review}
              onChange={(e) => setReview(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Paste or type your product review here..."
              rows={6}
              className={clsx(
                "w-full px-4 py-3 text-sm border rounded-xl resize-none",
                "text-slate-700 placeholder-slate-400",
                "focus:outline-none focus:ring-2 focus:ring-offset-0 transition-all",
                isOverLimit
                  ? "border-red-300 bg-red-50 focus:ring-red-400"
                  : "border-slate-200 bg-white focus:ring-indigo-500 focus:border-indigo-500"
              )}
            />
            {/* Character count */}
            <div className={clsx(
              "absolute bottom-3 right-3 text-xs font-mono tabular-nums",
              isOverLimit ? "text-red-500" : charCount > MAX_CHARS * 0.9 ? "text-amber-500" : "text-slate-300"
            )}>
              {charCount}/{MAX_CHARS}
            </div>
          </div>
          {isOverLimit && (
            <p className="mt-1 text-xs text-red-500">
              Review exceeds {MAX_CHARS} character limit.
            </p>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex items-center justify-between gap-3 pt-1">
          <button
            onClick={handleReset}
            disabled={!review && !result}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium
                       text-slate-500 hover:text-slate-700 border border-slate-200
                       rounded-lg hover:border-slate-300 disabled:opacity-40
                       disabled:cursor-not-allowed transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            Clear
          </button>

          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={clsx(
              "flex items-center gap-2 px-6 py-2.5 text-sm font-semibold",
              "rounded-lg transition-all duration-150",
              canSubmit
                ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm hover:shadow-md"
                : "bg-slate-100 text-slate-400 cursor-not-allowed"
            )}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Analyze Review
              </>
            )}
          </button>
        </div>

        <p className="text-xs text-slate-400 text-right">
          <Clock className="w-3 h-3 inline mr-0.5" />
          Press Ctrl+Enter to submit
        </p>
      </div>

      {/* Result card */}
      {result && (
        <ResultCard result={result} onReset={handleReset} />
      )}

      {/* Sample reviews */}
      {!result && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-amber-500" />
            <h3 className="text-sm font-semibold text-slate-700">Try a sample review</h3>
          </div>
          <div className="space-y-2">
            {SAMPLE_REVIEWS.map((sample, i) => (
              <button
                key={i}
                onClick={() => handleSample(sample)}
                className="w-full text-left px-4 py-3 rounded-lg text-sm text-slate-600
                           bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700
                           border border-slate-100 hover:border-indigo-200
                           transition-all line-clamp-2"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
