// 路由配置文件
import { Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import TrainingPage from './pages/Training';
import InferencePage from './pages/Inference';
import DataProcessingPage from './pages/DataProcessing';

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/data" replace />} />
        <Route path="data" element={<DataProcessingPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="inference" element={<InferencePage />} />
      </Route>
    </Routes>
  );
}

export default App;
