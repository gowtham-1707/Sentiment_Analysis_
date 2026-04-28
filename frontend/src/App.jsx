import React, { useState, useEffect, useCallback } from "react";
import { Toaster } from "react-hot-toast";
import {
  MessageSquare,
  Upload,
  BarChart3,
  Activity,
  Brain,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Github,
} from "lucide-react";
import clsx from "clsx";

import ReviewForm   from "./components/ReviewForm";
import BulkUpload   from "./components/BulkUpload";
import Dashboard    from "./components/Dashboard";
import { getHealth, getInfo } from "./api";
const TABS = [
  {
    id:    "single",
    label: "Single Review",
    icon:  MessageSquare,
    desc:  "Analyze one review at a time",
  },
  {
    id:    "bulk",
    label: "Bulk Upload",
    icon:  Upload,
    desc:  "Upload CSV for batch analysis",
  },
  {
    id:    "dashboard",
    label: "Analytics",
    icon:  BarChart3,
    desc:  "View sentiment insights",
  },
];

function StatusBadge({ status, label }) {
  const isHealthy = status === "healthy" || status === true;
  const isLoading = status === "loading";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
        isLoading && "bg-gray-100 text-gray-500",
        !isLoading && isHealthy  && "bg-emerald-50 text-emerald-700 border border-emerald-200",
        !isLoading && !isHealthy && "bg-red-50 text-red-700 border border-red-200"
      )}
    >
      {isLoading ? (
        <Loader2 className="w-3 h-3 animate-spin" />
      ) : isHealthy ? (
        <CheckCircle2 className="w-3 h-3" />
      ) : (
        <AlertCircle className="w-3 h-3" />
      )}
      {label}
    </span>
  );
}

export default function App() {
  const [activeTab,    setActiveTab]    = useState("single");
  const [apiStatus,    setApiStatus]    = useState("loading");
  const [modelLoaded,  setModelLoaded]  = useState("loading");
  const [modelVersion, setModelVersion] = useState(null);
  const [uptime,       setUptime]       = useState(null);
  const [bulkResults,  setBulkResults]  = useState(null);
  const checkHealth = useCallback(async () => {
    try {
      const [health, info] = await Promise.all([getHealth(), getInfo()]);
      setApiStatus(health.status);
      setModelLoaded(health.model_loaded);
      setModelVersion(info?.model?.model_version ?? "—");
      setUptime(info?.api?.uptime_seconds ?? null);
    } catch {
      setApiStatus("unhealthy");
      setModelLoaded(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);
  const handleBulkResults = (results) => {
    setBulkResults(results);
    setActiveTab("dashboard");
  };
  const formatUptime = (seconds) => {
    if (!seconds) return null;
    if (seconds < 60)   return `${Math.floor(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans">

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: { background: "#1e293b", color: "#f8fafc", fontSize: "14px" },
          success: { iconTheme: { primary: "#10b981", secondary: "#f8fafc" } },
          error:   { iconTheme: { primary: "#ef4444", secondary: "#f8fafc" } },
        }}
      />

      {/* ── HEADER ──────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">

          {/* Top bar: Logo + Status */}
          <div className="flex items-center justify-between h-16">

            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-indigo-600 rounded-lg flex items-center justify-center shadow-sm">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold text-slate-900 leading-tight">
                  Sentiment Analysis
                </h1>
                <p className="text-xs text-slate-400 leading-tight">
                  MLOps · Product Reviews
                </p>
              </div>
            </div>

            {/* Status row */}
            <div className="flex items-center gap-2 flex-wrap justify-end">
              <StatusBadge
                status={apiStatus === "healthy" ? true : apiStatus === "loading" ? "loading" : false}
                label="API"
              />
              <StatusBadge
                status={modelLoaded === "loading" ? "loading" : modelLoaded}
                label={
                  modelVersion && modelVersion !== "—"
                    ? `Model v${modelVersion}`
                    : "Model"
                }
              />
              {uptime && (
                <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1
                                 rounded-full text-xs font-medium bg-slate-100 text-slate-500">
                  <Activity className="w-3 h-3" />
                  Up {formatUptime(uptime)}
                </span>
              )}
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1
                           rounded-full text-xs font-medium bg-slate-100 text-slate-500
                           hover:bg-slate-200 hover:text-slate-700"
              >
                <Github className="w-3 h-3" />
                API Docs
              </a>
            </div>
          </div>

          {/* Navigation tabs */}
          <nav className="flex gap-1 -mb-px">
            {TABS.map((tab) => {
              const Icon    = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-3 text-sm font-medium",
                    "border-b-2 transition-all duration-150 whitespace-nowrap",
                    isActive
                      ? "border-indigo-600 text-indigo-600"
                      : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>

        </div>
      </header>

      {/* ── MAIN CONTENT ────────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">

        {/* API down banner */}
        {apiStatus !== "loading" && apiStatus !== "healthy" && (
          <div className="mb-6 flex items-center gap-3 px-4 py-3 bg-red-50
                          border border-red-200 rounded-lg text-red-700 text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <div>
              <span className="font-semibold">API Unavailable — </span>
              The backend is not responding. Ensure Docker containers are running:
              <code className="ml-1 px-1.5 py-0.5 bg-red-100 rounded text-xs font-mono">
                docker compose up
              </code>
            </div>
          </div>
        )}

        {/* Tab content */}
        <div>
          {activeTab === "single" && (
            <ReviewForm apiOnline={apiStatus === "healthy"} />
          )}
          {activeTab === "bulk" && (
            <BulkUpload
              apiOnline={apiStatus === "healthy"}
              onResults={handleBulkResults}
            />
          )}
          {activeTab === "dashboard" && (
            <Dashboard bulkResults={bulkResults} />
          )}
        </div>

      </main>

      {/* ── FOOTER ──────────────────────────────────────────── */}
      <footer className="mt-16 border-t border-slate-200 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6
                        flex flex-col sm:flex-row items-center justify-between
                        gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <Brain className="w-3.5 h-3.5 text-indigo-400" />
            <span>Sentiment Analysis MLOps — End-to-End AI Application</span>
          </div>
          <div className="flex items-center gap-4">
            <a href="http://localhost:5000"  target="_blank" rel="noreferrer"
               className="hover:text-indigo-500 transition-colors">MLflow</a>
            <a href="http://localhost:8080"  target="_blank" rel="noreferrer"
               className="hover:text-indigo-500 transition-colors">Airflow</a>
            <a href="http://localhost:3001"  target="_blank" rel="noreferrer"
               className="hover:text-indigo-500 transition-colors">Grafana</a>
            <a href="http://localhost:9090"  target="_blank" rel="noreferrer"
               className="hover:text-indigo-500 transition-colors">Prometheus</a>
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
               className="hover:text-indigo-500 transition-colors">API Docs</a>
          </div>
        </div>
      </footer>

    </div>
  );
}
