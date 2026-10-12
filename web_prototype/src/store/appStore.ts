import { create } from 'zustand';

// 训练数据点类型 (对应 results.txt 的 17 列)
export interface TrainingDataPoint {
  epoch: number;
  totalEpochs: number;
  gpuMemory: string;
  trainBoxLoss: number;
  trainObjLoss: number;
  trainClsLoss: number;
  trainTotalLoss: number;
  consLoss: number;
  labelsCount: number;
  imgSize: number;
  precision: number;
  recall: number;
  mAP50: number;
  mAP75: number;
  mAP5095: number;
  valBoxLoss: number;
  valObjLoss: number;
  valClsLoss: number;
}

// 训练状态
interface TrainingState {
  isTraining: boolean;
  historyData: TrainingDataPoint[];
  logs: string[];
  currentEpoch: number;
  totalEpochs: number;
  gpuMemory: string;
  currentMap50: number;
  currentMap75: number;
  currentMap5095: number;
  precision: number;
  recall: number;
  trainTotalLoss: number;
  valTotalLoss: number;
  consLoss: number;
}

interface AppState {
  isLiveMode: boolean;
  toggleMode: () => void;
  setMode: (mode: boolean) => void;
  
  // 训练状态
  training: TrainingState;
  setTraining: (state: Partial<TrainingState>) => void;
  addLog: (log: string) => void;
  addHistoryPoint: (point: TrainingDataPoint) => void;
  resetTraining: () => void;
}

const initialTrainingState: TrainingState = {
  isTraining: false,
  historyData: [],
  logs: [],
  currentEpoch: 0,
  totalEpochs: 200,
  gpuMemory: '0G',
  currentMap50: 0,
  currentMap75: 0,
  currentMap5095: 0,
  precision: 0,
  recall: 0,
  trainTotalLoss: 0,
  valTotalLoss: 0,
  consLoss: 0,
};

export const useAppStore = create<AppState>((set) => ({
  isLiveMode: false,
  toggleMode: () => set((state) => ({ isLiveMode: !state.isLiveMode })),
  setMode: (mode: boolean) => set({ isLiveMode: mode }),
  
  // 训练状态
  training: initialTrainingState,
  
  setTraining: (newState) => set((state) => ({
    training: { ...state.training, ...newState }
  })),
  
  addLog: (log) => set((state) => ({
    training: {
      ...state.training,
      logs: [...state.training.logs.slice(-500), log] // 保留最近 500 条日志，足够 200 epochs
    }
  })),
  
  addHistoryPoint: (point) => set((state) => ({
    training: {
      ...state.training,
      historyData: [...state.training.historyData, point],
      currentEpoch: point.epoch,
      totalEpochs: point.totalEpochs,
      gpuMemory: point.gpuMemory,
      currentMap50: point.mAP50,
      currentMap75: point.mAP75,
      currentMap5095: point.mAP5095,
      precision: point.precision,
      recall: point.recall,
      trainTotalLoss: point.trainTotalLoss,
      valTotalLoss: point.valBoxLoss + point.valObjLoss + point.valClsLoss,
      consLoss: point.consLoss,
    }
  })),
  
  resetTraining: () => set({ training: initialTrainingState }),
}));
