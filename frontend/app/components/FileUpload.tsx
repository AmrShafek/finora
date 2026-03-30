"use client";
import { useState, useRef } from "react";

const API = "http://127.0.0.1:8001";

interface UploadResult {
  date: string;
  amount: number;
  category: string;
  description: string;
}

export default function FileUpload({ onUploadSuccess }: { onUploadSuccess?: () => void }) {
  const [isDragging, setIsDragging]   = useState(false);
  const [loading, setLoading]         = useState(false);
  const [toast, setToast]             = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [result, setResult]           = useState<UploadResult | null>(null);
  const [preview, setPreview]         = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const showToast = (type: "success" | "error", message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  };

  const handleFile = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      showToast("error", "Please upload an image file (JPG, PNG, or WebP)");
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API}/upload-doc`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setResult(data.data);
      showToast("success", "Document processed and saved to database!");
      if (onUploadSuccess) onUploadSuccess();
    } catch (err: any) {
      showToast("error", err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div style={{
      background: "rgba(10,20,35,0.8)",
      border: "1px solid rgba(0,228,160,0.15)",
      borderRadius: 20,
      padding: "28px 28px",
      fontFamily: "'DM Sans', sans-serif",
      color: "#f0f8ff",
      position: "relative",
    }}>

      {/* Toast notification */}
      {toast && (
        <div style={{
          position: "absolute",
          top: 16, right: 16,
          background: toast.type === "success" ? "rgba(0,228,160,0.15)" : "rgba(255,80,80,0.15)",
          border: `1px solid ${toast.type === "success" ? "rgba(0,228,160,0.4)" : "rgba(255,80,80,0.4)"}`,
          borderRadius: 12,
          padding: "12px 18px",
          fontSize: 13,
          color: toast.type === "success" ? "#00e4a0" : "#ff6b6b",
          display: "flex",
          alignItems: "center",
          gap: 8,
          maxWidth: 320,
          zIndex: 10,
        }}>
          <span>{toast.type === "success" ? "✓" : "✕"}</span>
          {toast.message}
        </div>
      )}

      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, letterSpacing: 4, textTransform: "uppercase", color: "rgba(0,228,160,0.7)", marginBottom: 4 }}>
          OCR Document Scanner
        </div>
        <div style={{ fontSize: 20, fontWeight: 700 }}>Upload Invoice or Receipt</div>
        <div style={{ fontSize: 13, color: "rgba(180,210,200,0.5)", marginTop: 4 }}>
          Finora will extract the data and save it to your expenses
        </div>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${isDragging ? "#00e4a0" : "rgba(0,228,160,0.25)"}`,
          borderRadius: 16,
          padding: "32px 24px",
          textAlign: "center",
          cursor: loading ? "not-allowed" : "pointer",
          background: isDragging ? "rgba(0,228,160,0.05)" : "rgba(255,255,255,0.02)",
          transition: "all 0.2s",
          marginBottom: 20,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          style={{ display: "none" }}
          disabled={loading}
        />

        {loading ? (
          <div>
            <div style={{
              width: 40, height: 40, border: "3px solid rgba(0,228,160,0.2)",
              borderTop: "3px solid #00e4a0", borderRadius: "50%",
              margin: "0 auto 16px",
              animation: "spin 1s linear infinite",
            }} />
            <div style={{ fontSize: 15, color: "#00e4a0" }}>Finora is analyzing the document...</div>
            <div style={{ fontSize: 12, color: "rgba(180,210,200,0.4)", marginTop: 6 }}>
              Extracting date, amount, category, and description
            </div>
          </div>
        ) : preview ? (
          <div>
            <img
              src={preview}
              alt="Document preview"
              style={{ maxHeight: 160, maxWidth: "100%", borderRadius: 8, marginBottom: 12, objectFit: "contain" }}
            />
            <div style={{ fontSize: 13, color: "rgba(0,228,160,0.7)" }}>Click to upload a different document</div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 40, marginBottom: 12 }}>📄</div>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>Drop your invoice or receipt here</div>
            <div style={{ fontSize: 13, color: "rgba(180,210,200,0.5)" }}>or click to browse — JPG, PNG, WebP</div>
          </div>
        )}
      </div>

      {/* Extracted data result */}
      {result && (
        <div style={{
          background: "rgba(0,228,160,0.07)",
          border: "1px solid rgba(0,228,160,0.2)",
          borderRadius: 14,
          padding: "18px 20px",
        }}>
          <div style={{ fontSize: 12, letterSpacing: 3, textTransform: "uppercase", color: "rgba(0,228,160,0.7)", marginBottom: 14 }}>
            Extracted & Saved
          </div>
          {[
            ["Date",        result.date],
            ["Amount",      `$${result.amount.toLocaleString()}`],
            ["Category",    result.category],
            ["Description", result.description],
          ].map(([label, value]) => (
            <div key={label} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "8px 0",
              borderBottom: "1px solid rgba(0,228,160,0.08)",
            }}>
              <span style={{ fontSize: 12, color: "rgba(180,210,200,0.5)", letterSpacing: 1 }}>{label}</span>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#f0f8ff" }}>{value}</span>
            </div>
          ))}
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
