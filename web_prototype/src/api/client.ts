import axios from 'axios';
import { useAppStore } from '../store/appStore';

const API_BASE = 'http://localhost:8000';

// 推理参数接口
export interface InferenceParams {
  file: File;
  weights: string;
  dataset: 'cityscapes' | 'kitti' | 'voc';
  confThres: number;
  imgSize: number;
}

// 推理结果接口
export interface InferenceResult {
  success: boolean;
  inferenceTime: number;
  fps: number;
  modelParams: string;
  objects: Array<{
    class: string;
    class_id: number;
    conf: number;
    bbox: number[];
  }>;
  numObjects: number;
  resultImageBase64: string;
}

// 数据集信息接口
export interface DatasetInfo {
  id: string;
  name: string;
  classes: string[];
  num_classes: number;
}

// 模型信息接口
export interface ModelInfo {
  name: string;
  path: string;
  size: number;
}

// Mock 推理结果
const mockInferenceResult: Omit<InferenceResult, 'resultImageBase64'> = {
  success: true,
  inferenceTime: 24.5,
  fps: 40.8,
  modelParams: '47.2M',
  objects: [
    { class: 'car', class_id: 2, conf: 0.95, bbox: [100, 150, 300, 350] },
    { class: 'person', class_id: 4, conf: 0.88, bbox: [400, 200, 480, 450] },
    { class: 'bicycle', class_id: 1, conf: 0.76, bbox: [50, 300, 120, 400] },
  ],
  numObjects: 3
};

// 统一 API 客户端
export const apiClient = {
  // 获取训练日志 (Results.txt)
  getTrainingLogs: async (resultsFile?: string) => {
    const isLive = useAppStore.getState().isLiveMode;
    if (isLive) {
      try {
        const params = resultsFile ? { results_file: resultsFile } : {};
        const res = await axios.get(`${API_BASE}/train/logs`, { params });
        return res.data;
      } catch (e) {
        console.error("Live API failed", e);
        return [];
      }
    }
    return [];
  },

  // 执行推理
  runInference: async (params: InferenceParams): Promise<InferenceResult> => {
    const isLive = useAppStore.getState().isLiveMode;
    
    if (isLive) {
      try {
        const formData = new FormData();
        formData.append('file', params.file);
        formData.append('weights', params.weights);
        formData.append('dataset', params.dataset);
        formData.append('conf_thres', params.confThres.toString());
        formData.append('img_size', params.imgSize.toString());
        
        const res = await axios.post<InferenceResult>(`${API_BASE}/inference`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000 // 60秒超时
        });
        return res.data;
      } catch (e) {
        console.error("Inference failed", e);
        throw e;
      }
    }

    // Mock 模式：模拟延迟并返回模拟结果
    await new Promise(r => setTimeout(r, 800 + Math.random() * 400));
    
    return {
      ...mockInferenceResult,
      resultImageBase64: '' // Mock 模式不返回图片
    };
  },

  // 获取可用模型列表
  getModels: async (): Promise<ModelInfo[]> => {
    const isLive = useAppStore.getState().isLiveMode;
    if (isLive) {
      try {
        const res = await axios.get(`${API_BASE}/models`);
        return res.data.models || [];
      } catch (e) {
        console.error("Failed to get models", e);
        return [];
      }
    }
    // 返回预设模型（三种数据集对应的最优模型）
    return [
      { name: 'C2F - Cityscapes→Foggy', path: '/home/yu/zjh/SSDA-YOLO-copy/runs/train/12_21_bestpt_maskratio_03.6/weights/best_student.pt', size: 180 },
      { name: 'V2C - VOC→Clipart', path: '/home/yu/zjh/SSDA-YOLO-copy/runs/train/A_3_11_V2C/weights/best_student.pt', size: 180 },
      { name: 'K2C - KITTI→CityScapes', path: '/home/yu/zjh/SSDA-YOLO-copy/runs/train/3_9_K2C_train3/weights/best_student.pt', size: 180 },
    ];
  },

  // 获取支持的数据集列表
  getDatasets: async (): Promise<DatasetInfo[]> => {
    const isLive = useAppStore.getState().isLiveMode;
    if (isLive) {
      try {
        const res = await axios.get(`${API_BASE}/datasets`);
        return res.data.datasets || [];
      } catch (e) {
        console.error("Failed to get datasets", e);
      }
    }
    // 返回预设数据集
    return [
      { id: 'cityscapes', name: 'Cityscapes → Foggy', classes: ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'], num_classes: 8 },
      { id: 'kitti', name: 'KITTI → CityScapes', classes: ['background', 'car'], num_classes: 2 },
      { id: 'voc', name: 'VOC → Clipart', classes: ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], num_classes: 21 },
    ];
  },

  // 下载模型
  downloadModel: (weights?: string) => {
    const isLive = useAppStore.getState().isLiveMode;
    if (isLive) {
      const url = weights 
        ? `${API_BASE}/model/download?weights=${encodeURIComponent(weights)}`
        : `${API_BASE}/model/download`;
      window.open(url, '_blank');
    } else {
      alert(`最优模型路径:\n${weights || '/home/yu/zjh/SSDA-YOLO-copy/runs/train/exp/weights/best_student.pt'}\n\n请使用该路径进行模型部署。`);
    }
  }
};
