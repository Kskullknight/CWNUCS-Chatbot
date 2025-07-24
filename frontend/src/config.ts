// API 설정
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';
export const API_ENDPOINTS = {
  chat: `${API_BASE_URL}/api/chat`,
  health: `${API_BASE_URL}/api/health`,
};