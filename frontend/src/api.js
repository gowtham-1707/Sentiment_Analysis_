import axios from "axios";
const BASE_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,                         
  headers: { "Content-Type": "application/json" },
});
api.interceptors.request.use(
  (config) => {
    console.debug(`→ ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      "An unexpected error occurred";
    return Promise.reject(new Error(message));
  }
);

export const predictSingle = async (review, productId = null) => {
  const { data } = await api.post("/api/v1/predict", {
    review,
    product_id: productId,
  });
  return data;
};

export const predictBulk = async (reviews) => {
  const { data } = await api.post("/api/v1/predict/bulk", { reviews });
  return data;
};

export const predictCSV = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/api/v1/predict/csv", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
  });
  return data;
};

export const getHealth = async () => {
  const { data } = await api.get("/health");
  return data;
};

export const getInfo = async () => {
  const { data } = await api.get("/info");
  return data;
};

export default api;