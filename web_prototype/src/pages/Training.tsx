import React, { useState, useEffect, useRef } from 'react';
import { Row, Col, Card, Form, Input, InputNumber, Slider, Switch, Button, Statistic, message, Progress, Tag, Select, Descriptions } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, CloudDownloadOutlined, ThunderboltOutlined, SettingOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { useAppStore } from '../store/appStore';
import type { TrainingDataPoint } from '../store/appStore';
import { apiClient } from '../api/client';
import { getMockTrainingData } from '../utils/mockData';

const TrainingPage: React.FC = () => {
  const [form] = Form.useForm();
  const isLive = useAppStore(state => state.isLiveMode);
  
  // 使用全局状态
  const training = useAppStore(state => state.training);
  const setTraining = useAppStore(state => state.setTraining);
  const addLogToStore = useAppStore(state => state.addLog);
  const addHistoryPoint = useAppStore(state => state.addHistoryPoint);
  const resetTraining = useAppStore(state => state.resetTraining);
  
  // 本地 loading 状态用于控制 UI
  const [loading, setLoading] = useState(training.isTraining);
  
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // 同步 loading 状态
  useEffect(() => {
    setTraining({ isTraining: loading });
  }, [loading, setTraining]);

  // 只在训练进行中自动滚动到日志底部
  useEffect(() => {
    if (logContainerRef.current && loading) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [training.logs, loading]);

  const addLog = (msg: string) => {
    addLogToStore(msg);
  };

  const startTraining = async () => {
    setLoading(true);
    resetTraining();
    setTraining({ isTraining: true });
    
    const formValues = form.getFieldsValue();
    const weightsPath = formValues.weights || '/home/yu/zjh/SSDA-YOLO-copy/weights/yolov5l.pt';
    const dataPath = formValues.data || '/home/yu/zjh/SSDA-YOLO-copy/data/yamls_sda/cityscapes_csfoggy_VOC.yaml';
    const imgSize = formValues.imgSize || 960;
    
    // 初始化日志 - 模拟真实训练启动
    const initLogs = [
      `train: weights=${weightsPath}`,
      `       data=${dataPath}`,
      `       cfg=/home/yu/zjh/SSDA-YOLO-copy/models/yolov5l_city_seg.yaml`,
      `hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005`,
      ``,
      `                 from  n    params  module                                  arguments`,
      `  0                -1  1      7040  models.common.Conv                      [3, 64, 6, 2, 2]`,
      `  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]`,
      `  ...`,
      ` 24                -1  1    525312  Detect                                  [8, [[10, 13, 16, 30], ...]]`,
      `Model Summary: 476 layers, 46,563,709 parameters, 46,563,709 gradients`,
      ``,
      `Scaled weight_decay = 0.0005`,
      `Optimizer groups: 123 .bias, 130 conv.weight, 127 other`,
      ``,
      `train_source_real: Scanning '/home/yu/zjh/datasets/cityscapes/labels/train'...`,
      `train_target_real: Scanning '/home/yu/zjh/datasets/foggy_cityscapes/labels/train'...`,
      ``,
      `Image sizes ${imgSize} train, ${imgSize} test`,
      `Using 4 dataloader_sr workers`,
      `Starting training for ${formValues.epochs || 200} epochs...`,
      ``
    ];
    initLogs.forEach(log => addLog(log));
    
    message.loading({ content: '正在初始化...', key: 'train', duration: 1 });

    if (isLive) {
      try {
        const logsData = await apiClient.getTrainingLogs();
        if (logsData && logsData.length > 0) {
          let i = 0;
          timerRef.current = setInterval(() => {
            if (i >= logsData.length) {
              clearInterval(timerRef.current!);
              setLoading(false);
              message.success({ content: '训练回放完成!', key: 'train' });
              return;
            }
            updateState(logsData[i]);
            i += 3;
          }, 80);
        } else {
          message.warning('未找到训练日志');
          setLoading(false);
        }
      } catch {
        message.error('连接后端失败');
        setLoading(false);
      }
    } else {
      // Mock 模式：使用真实的 results.txt 训练数据
      const mockData = getMockTrainingData();
      
      let i = 0;
      timerRef.current = setInterval(() => {
        if (i >= mockData.length) {
          clearInterval(timerRef.current!);
          setLoading(false);
          const finalData = mockData[mockData.length - 1];
          // 计算训练过程中的最高 mAP
          const bestMap50 = Math.max(...mockData.map(d => d.mAP50));
          const bestMap5095 = Math.max(...mockData.map(d => d.mAP5095));
          addLog(``);
          addLog(`${finalData.totalEpochs} epochs completed in 28.234 hours.`);
          addLog(`Results saved to runs/train/${formValues.name || 'exp'}`);
          addLog(`Best mAP@0.5 = ${bestMap50.toFixed(4)}, mAP@0.5:0.95 = ${bestMap5095.toFixed(4)}`);
          message.success({ content: '训练完成!', key: 'train' });
          return;
        }
        
        const point = mockData[i];
        if (point) {
          updateState(point);
          
          // 输出训练日志 - 模拟真实 YOLO 训练输出格式
          const epochStr = `${point.epoch - 1}/${point.totalEpochs - 1}`.padStart(10);
          const memStr = point.gpuMemory.padStart(8);
          const boxStr = point.trainBoxLoss.toFixed(4).padStart(10);
          const objStr = point.trainObjLoss.toFixed(4).padStart(10);
          const clsStr = point.trainClsLoss.toFixed(4).padStart(10);
          const totalStr = point.trainTotalLoss.toFixed(4).padStart(10);
          const consStr = point.consLoss.toExponential(2).padStart(10);
          const labelsStr = String(point.labelsCount).padStart(10);
          const imgStr = String(point.imgSize).padStart(10);
          
          addLog(`${epochStr}${memStr}${boxStr}${objStr}${clsStr}${totalStr}${consStr}${labelsStr}${imgStr}`);
          
          // 每个 epoch 结束后显示验证指标
          if (i > 0) {
            const pStr = point.precision.toFixed(4).padStart(10);
            const rStr = point.recall.toFixed(4).padStart(10);
            const map50Str = point.mAP50.toFixed(4).padStart(10);
            const map75Str = point.mAP75.toFixed(4).padStart(10);
            const map5095Str = point.mAP5095.toFixed(4).padStart(10);
            const vBoxStr = point.valBoxLoss.toFixed(4).padStart(10);
            const vObjStr = point.valObjLoss.toFixed(4).padStart(10);
            const vClsStr = point.valClsLoss.toFixed(4).padStart(10);
            addLog(`                    val:${pStr}${rStr}${map50Str}${map75Str}${map5095Str}${vBoxStr}${vObjStr}${vClsStr}`);
          }
        }
        i++;
      }, 150); // 每150ms更新一次，让演示更流畅
    }
  };

  const stopTraining = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setLoading(false);
    addLog('');
    addLog('Training interrupted by user.');
    message.info('训练已停止');
  };

  const updateState = (point: TrainingDataPoint) => {
    addHistoryPoint(point);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Train Loss 曲线配置
  const trainLossOption = {
    title: { text: 'Train Loss', left: 'center', top: 0, textStyle: { fontSize: 14, fontWeight: 500 } },
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(50,50,50,0.9)', textStyle: { color: '#fff' } },
    legend: { top: 25, itemGap: 12, data: ['Box', 'Obj', 'Cls'], textStyle: { fontSize: 11 } },
    grid: { left: 45, right: 15, top: 60, bottom: 25 },
    xAxis: { type: 'category', data: training.historyData.map(d => d.epoch), axisLabel: { fontSize: 10 }, name: 'Epoch' },
    yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
    series: [
      { name: 'Box', type: 'line', smooth: true, data: training.historyData.map(d => d.trainBoxLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#1890ff' } },
      { name: 'Obj', type: 'line', smooth: true, data: training.historyData.map(d => d.trainObjLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#fa541c' } },
      { name: 'Cls', type: 'line', smooth: true, data: training.historyData.map(d => d.trainClsLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#722ed1' } }
    ]
  };

  // Val Loss 曲线配置
  const valLossOption = {
    title: { text: 'Val Loss', left: 'center', top: 0, textStyle: { fontSize: 14, fontWeight: 500 } },
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(50,50,50,0.9)', textStyle: { color: '#fff' } },
    legend: { top: 25, itemGap: 12, data: ['Box', 'Obj', 'Cls'], textStyle: { fontSize: 11 } },
    grid: { left: 45, right: 15, top: 60, bottom: 25 },
    xAxis: { type: 'category', data: training.historyData.map(d => d.epoch), axisLabel: { fontSize: 10 }, name: 'Epoch' },
    yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
    series: [
      { name: 'Box', type: 'line', smooth: true, data: training.historyData.map(d => d.valBoxLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#13c2c2' } },
      { name: 'Obj', type: 'line', smooth: true, data: training.historyData.map(d => d.valObjLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#eb2f96' } },
      { name: 'Cls', type: 'line', smooth: true, data: training.historyData.map(d => d.valClsLoss), showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#faad14' } }
    ]
  };

  // mAP 曲线配置 - 显示更多指标
  const mapOption = {
    title: { text: 'mAP & Metrics', left: 'center', top: 0, textStyle: { fontSize: 13, fontWeight: 500 } },
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(50,50,50,0.9)', textStyle: { color: '#fff', fontSize: 11 } },
    legend: { top: 20, itemGap: 6, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 9 } },
    grid: { left: 40, right: 10, top: 45, bottom: 22 },
    xAxis: { type: 'category', data: training.historyData.map(d => d.epoch), axisLabel: { fontSize: 9 }, name: 'Epoch', nameTextStyle: { fontSize: 9 } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { fontSize: 9 } },
    series: [
      { name: 'mAP@.5', type: 'line', smooth: true, data: training.historyData.map(d => d.mAP50), showSymbol: false, itemStyle: { color: '#52c41a' }, lineStyle: { width: 2 } },
      { name: 'mAP@.75', type: 'line', smooth: true, data: training.historyData.map(d => d.mAP75), showSymbol: false, itemStyle: { color: '#1890ff' }, lineStyle: { width: 2 } },
      { name: 'mAP@.5:.95', type: 'line', smooth: true, data: training.historyData.map(d => d.mAP5095), showSymbol: false, itemStyle: { color: '#faad14' }, lineStyle: { width: 2 } },
      { name: 'P', type: 'line', smooth: true, data: training.historyData.map(d => d.precision), showSymbol: false, itemStyle: { color: '#eb2f96' }, lineStyle: { width: 1.5, type: 'dashed' } },
      { name: 'R', type: 'line', smooth: true, data: training.historyData.map(d => d.recall), showSymbol: false, itemStyle: { color: '#722ed1' }, lineStyle: { width: 1.5, type: 'dashed' } }
    ]
  };

  // 计算最佳值
  const bestMap50 = training.historyData.length > 0 ? Math.max(...training.historyData.map(d => d.mAP50)) : 0;
  const bestMap5095 = training.historyData.length > 0 ? Math.max(...training.historyData.map(d => d.mAP5095)) : 0;
  const bestPrecision = training.historyData.length > 0 ? Math.max(...training.historyData.map(d => d.precision)) : 0;
  const bestRecall = training.historyData.length > 0 ? Math.max(...training.historyData.map(d => d.recall)) : 0;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <Row gutter={16} style={{ flex: 1, minHeight: 0 }}>
        {/* 左侧配置面板 */}
        <Col span={6} style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <Card 
            title={<><ThunderboltOutlined /> 实验配置</>} 
            size="small" 
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'auto', paddingRight: 8 } }}
          >
            <Form 
              form={form} 
              layout="vertical" 
              size="small" 
              initialValues={{ 
                data: '/home/yu/zjh/SSDA-YOLO-copy/data/yamls_sda/cityscapes_csfoggy_VOC.yaml',
                weights: '/home/yu/zjh/SSDA-YOLO-copy/weights/yolov5l.pt', 
                epochs: 200, 
                batch: 2,
                imgSize: 960,
                device: '0',
                name: 'exp',
                lambda: 0.005, 
                alpha: 2.0,
                teacherAlpha: 0.99,
                maskRatio: 0.6,
                maskBlockSize: 64,
                maskColorJitterS: 0.2,
                maskColorJitterP: 0.2,
                useDPLA: true, 
                useDAAT: true,
                useConsistencyLoss: true
              }}
            >
              {/* 数据集路径 */}
              <Form.Item label="数据集配置" name="data" style={{ marginBottom: 6 }}>
                <Input placeholder="dataset yaml path" style={{ fontSize: 11 }} />
              </Form.Item>
              
              {/* 预训练权重 */}
              <Form.Item label="预训练权重" name="weights" style={{ marginBottom: 6 }}>
                <Input placeholder="/path/to/weights.pt" style={{ fontSize: 11 }} />
              </Form.Item>
              
              {/* 输出路径 */}
              <Form.Item label="输出名称" name="name" style={{ marginBottom: 6 }}>
                <Input placeholder="实验名称，保存到 runs/train/{name}" />
              </Form.Item>

              {/* Epochs, Batch, ImgSize, Device */}
              <Row gutter={6}>
                <Col span={6}><Form.Item label="Epochs" name="epochs" style={{ marginBottom: 6 }}><InputNumber style={{ width: '100%' }} min={1} max={1000} /></Form.Item></Col>
                <Col span={6}><Form.Item label="Batch" name="batch" style={{ marginBottom: 6 }}><InputNumber style={{ width: '100%' }} min={1} max={64} /></Form.Item></Col>
                <Col span={6}><Form.Item label="ImgSize" name="imgSize" style={{ marginBottom: 6 }}><InputNumber style={{ width: '100%' }} min={320} max={1280} step={32} /></Form.Item></Col>
                <Col span={6}>
                  <Form.Item label="Device" name="device" style={{ marginBottom: 6 }}>
                    <Select size="small" options={[
                      { value: '0', label: 'GPU:0' },
                      { value: '0,1', label: 'GPU:0,1' },
                      { value: 'cpu', label: 'CPU' }
                    ]} />
                  </Form.Item>
                </Col>
              </Row>
              
              {/* 蒸馏系数 λ */}
              <Form.Item label="蒸馏系数 λ" style={{ marginBottom: 6 }}>
                <Row gutter={6} align="middle">
                  <Col flex="auto">
                    <Form.Item name="lambda" noStyle>
                      <Slider min={0} max={0.2} step={0.005} tooltip={{ formatter: (val) => val?.toFixed(3) }} />
                    </Form.Item>
                  </Col>
                  <Col flex="55px">
                    <Form.Item name="lambda" noStyle>
                      <InputNumber min={0} max={0.2} step={0.005} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              </Form.Item>

              {/* 一致性损失系数 α */}
              <Form.Item label="一致性损失系数 α" style={{ marginBottom: 6 }}>
                <Row gutter={6} align="middle">
                  <Col flex="auto">
                    <Form.Item name="alpha" noStyle>
                      <Slider min={0} max={5} step={0.1} tooltip={{ formatter: (val) => val?.toFixed(1) }} />
                    </Form.Item>
                  </Col>
                  <Col flex="55px">
                    <Form.Item name="alpha" noStyle>
                      <InputNumber min={0} max={5} step={0.1} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              </Form.Item>

              {/* Teacher EMA Alpha */}
              <Form.Item label="Teacher EMA α" style={{ marginBottom: 6 }}>
                <Row gutter={6} align="middle">
                  <Col flex="auto">
                    <Form.Item name="teacherAlpha" noStyle>
                      <Slider min={0.9} max={0.999} step={0.001} tooltip={{ formatter: (val) => val?.toFixed(3) }} />
                    </Form.Item>
                  </Col>
                  <Col flex="55px">
                    <Form.Item name="teacherAlpha" noStyle>
                      <InputNumber min={0.9} max={0.999} step={0.001} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              </Form.Item>

              {/* Masking Ratio */}
              <Form.Item label="Masking Ratio" style={{ marginBottom: 6 }}>
                <Row gutter={6} align="middle">
                  <Col flex="auto">
                    <Form.Item name="maskRatio" noStyle>
                      <Slider min={0} max={1} step={0.05} tooltip={{ formatter: (val) => val?.toFixed(2) }} />
                    </Form.Item>
                  </Col>
                  <Col flex="55px">
                    <Form.Item name="maskRatio" noStyle>
                      <InputNumber min={0} max={1} step={0.05} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              </Form.Item>

              {/* Mask Block Size */}
              <Form.Item label="Mask Block Size" style={{ marginBottom: 6 }}>
                <Row gutter={6} align="middle">
                  <Col flex="auto">
                    <Form.Item name="maskBlockSize" noStyle>
                      <Slider min={16} max={128} step={16} tooltip={{ formatter: (val) => `${val}px` }} />
                    </Form.Item>
                  </Col>
                  <Col flex="55px">
                    <Form.Item name="maskBlockSize" noStyle>
                      <InputNumber min={16} max={128} step={16} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                </Row>
              </Form.Item>

              {/* Color Jitter 参数 */}
              <Row gutter={6}>
                <Col span={12}>
                  <Form.Item label="Color Jitter S" tooltip="色彩抖动强度" style={{ marginBottom: 6 }}>
                    <Form.Item name="maskColorJitterS" noStyle>
                      <InputNumber min={0} max={1} step={0.05} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Color Jitter P" tooltip="色彩抖动概率" style={{ marginBottom: 6 }}>
                    <Form.Item name="maskColorJitterP" noStyle>
                      <InputNumber min={0} max={1} step={0.05} size="small" style={{ width: '100%' }} />
                    </Form.Item>
                  </Form.Item>
                </Col>
              </Row>

              {/* 开关选项 */}
              <Row gutter={6} style={{ marginTop: 2 }}>
                <Col span={8}><Form.Item label="DPLA" name="useDPLA" valuePropName="checked" style={{ marginBottom: 0 }}><Switch size="small" /></Form.Item></Col>
                <Col span={8}><Form.Item label="DAAT" name="useDAAT" valuePropName="checked" style={{ marginBottom: 0 }}><Switch size="small" /></Form.Item></Col>
                <Col span={8}><Form.Item label="一致性" name="useConsistencyLoss" valuePropName="checked" style={{ marginBottom: 0 }}><Switch size="small" /></Form.Item></Col>
              </Row>
            </Form>
            <div style={{ marginTop: 'auto', paddingTop: 8 }}>
              {!loading ? (
                <Button type="primary" icon={<PlayCircleOutlined />} block size="large" onClick={startTraining}>开始训练</Button>
              ) : (
                <Button danger icon={<PauseCircleOutlined />} block size="large" onClick={stopTraining}>停止训练</Button>
              )}
              <Button icon={<CloudDownloadOutlined />} block style={{ marginTop: 6 }} onClick={() => apiClient.downloadModel()} disabled={training.historyData.length === 0}>下载最优模型</Button>
            </div>
          </Card>
        </Col>
        
        {/* 右侧主内容区 */}
        <Col span={18} style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
          {/* 统计卡片 - 显示更多指标 */}
          <Row gutter={8} style={{ marginBottom: 12, flexShrink: 0 }}>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic 
                  title={<span style={{ fontSize: 10 }}>Epoch</span>} 
                  value={training.currentEpoch} 
                  suffix={<span style={{ fontSize: 11, color: '#999' }}>/ {training.totalEpochs}</span>} 
                  valueStyle={{ color: '#1890ff', fontSize: 16 }} 
                />
                <Progress percent={Math.round((training.currentEpoch / training.totalEpochs) * 100)} size="small" showInfo={false} strokeColor="#1890ff" style={{ marginTop: 2 }} />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic title={<span style={{ fontSize: 10 }}>mAP@0.5</span>} value={training.currentMap50} precision={4} valueStyle={{ color: '#52c41a', fontSize: 16 }} />
                <Tag color="green" style={{ marginTop: 2, fontSize: 9 }}>Best: {bestMap50.toFixed(4)}</Tag>
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic title={<span style={{ fontSize: 10 }}>mAP@0.5:0.95</span>} value={training.currentMap5095} precision={4} valueStyle={{ color: '#faad14', fontSize: 16 }} />
                <Tag color="orange" style={{ marginTop: 2, fontSize: 9 }}>Best: {bestMap5095.toFixed(4)}</Tag>
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic title={<span style={{ fontSize: 10 }}>Precision</span>} value={training.precision} precision={4} valueStyle={{ color: '#eb2f96', fontSize: 16 }} />
                <Tag color="magenta" style={{ marginTop: 2, fontSize: 9 }}>Best: {bestPrecision.toFixed(4)}</Tag>
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic title={<span style={{ fontSize: 10 }}>Recall</span>} value={training.recall} precision={4} valueStyle={{ color: '#722ed1', fontSize: 16 }} />
                <Tag color="purple" style={{ marginTop: 2, fontSize: 9 }}>Best: {bestRecall.toFixed(4)}</Tag>
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" styles={{ body: { padding: '6px 10px' } }}>
                <Statistic title={<span style={{ fontSize: 10 }}>GPU Memory</span>} value={training.gpuMemory} valueStyle={{ color: '#1890ff', fontSize: 16 }} />
                <Tag color="blue" style={{ marginTop: 2, fontSize: 9 }}>Loss: {training.trainTotalLoss.toFixed(2)}</Tag>
              </Card>
            </Col>
          </Row>
          
          {/* 三个图表：Train Loss, Val Loss, mAP */}
          <Row gutter={12} style={{ marginBottom: 12, flexShrink: 0 }}>
            <Col span={8}><Card size="small" styles={{ body: { padding: '8px 4px' } }}><ReactECharts option={trainLossOption} style={{ height: 200 }} /></Card></Col>
            <Col span={8}><Card size="small" styles={{ body: { padding: '8px 4px' } }}><ReactECharts option={valLossOption} style={{ height: 200 }} /></Card></Col>
            <Col span={8}><Card size="small" styles={{ body: { padding: '8px 4px' } }}><ReactECharts option={mapOption} style={{ height: 200 }} /></Card></Col>
          </Row>
          
          {/* 训练日志和实时指标 */}
          <Row gutter={12} style={{ flex: 1, minHeight: 0 }}>
          {/* 训练日志 */}
            <Col span={16} style={{ height: '100%' }}>
              <Card 
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>训练日志</span>
                    <span style={{ fontSize: 10, color: '#999', fontWeight: 'normal', fontFamily: 'monospace' }}>
                      Epoch | GPU | box | obj | cls | total | cons | labels | img
                    </span>
                  </div>
                }
                size="small" 
                style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
                styles={{ body: { padding: 0, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' } }}
              >
                <div 
                  ref={logContainerRef}
                  className="terminal-output" 
                  style={{ flex: 1, minHeight: 100, overflowY: 'auto', fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace", fontSize: 11 }}
                >
                  {training.logs.map((log, i) => (
                    <div 
                      key={i} 
                      className="log-line" 
                      style={{ 
                        whiteSpace: 'pre',
                        color: log.includes('val:') ? '#4ec9b0' : 
                               log.includes('completed') || log.includes('Best') ? '#52c41a' : 
                               log.includes('interrupted') ? '#ff4d4f' : '#d4d4d4'
                      }}
                    >
                      {log}
                    </div>
                  ))}
                </div>
              </Card>
            </Col>
            
            {/* 右侧实时指标面板 */}
            <Col span={8} style={{ height: '100%' }}>
              <Card 
                title={<><SettingOutlined /> 训练指标</>}
                size="small" 
                style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
                styles={{ body: { flex: 1, overflow: 'auto', padding: '12px' } }}
              >
                <Descriptions column={1} size="small" labelStyle={{ fontSize: 11, color: '#999' }} contentStyle={{ fontSize: 12, fontWeight: 500 }}>
                  <Descriptions.Item label="当前 Epoch">
                    <span style={{ color: '#1890ff' }}>{training.currentEpoch}</span> / {training.totalEpochs}
                  </Descriptions.Item>
                  <Descriptions.Item label="进度">
                    <Progress 
                      percent={Math.round((training.currentEpoch / training.totalEpochs) * 100)} 
                      size="small" 
                      strokeColor="#1890ff"
                      style={{ width: 120 }}
                    />
                  </Descriptions.Item>
                </Descriptions>
                
                <div style={{ marginTop: 12, marginBottom: 8, fontSize: 11, color: '#999', borderBottom: '1px solid #303030', paddingBottom: 4 }}>
                  最佳指标 (Best)
                </div>
                <Descriptions column={1} size="small" labelStyle={{ fontSize: 11, color: '#999' }} contentStyle={{ fontSize: 12 }}>
                  <Descriptions.Item label="mAP@0.5">
                    <span style={{ color: '#52c41a', fontWeight: 600 }}>{bestMap50.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="mAP@0.5:0.95">
                    <span style={{ color: '#faad14', fontWeight: 600 }}>{bestMap5095.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Precision">
                    <span style={{ color: '#eb2f96', fontWeight: 600 }}>{bestPrecision.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Recall">
                    <span style={{ color: '#722ed1', fontWeight: 600 }}>{bestRecall.toFixed(4)}</span>
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 12, marginBottom: 8, fontSize: 11, color: '#999', borderBottom: '1px solid #303030', paddingBottom: 4 }}>
                  当前 Epoch 指标
                </div>
                <Descriptions column={2} size="small" labelStyle={{ fontSize: 11, color: '#999' }} contentStyle={{ fontSize: 12 }}>
                  <Descriptions.Item label="Precision">
                    <span style={{ color: '#eb2f96' }}>{training.precision.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Recall">
                    <span style={{ color: '#722ed1' }}>{training.recall.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Train Loss">
                    <span style={{ color: '#fa541c' }}>{training.trainTotalLoss.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Val Loss">
                    <span style={{ color: '#13c2c2' }}>{training.valTotalLoss.toFixed(4)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="Cons Loss">
                    <span style={{ color: '#2db7f5' }}>{training.consLoss.toExponential(2)}</span>
                  </Descriptions.Item>
                  <Descriptions.Item label="GPU Mem">
                    <span style={{ color: '#722ed1' }}>{training.gpuMemory}</span>
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 12, marginBottom: 8, fontSize: 11, color: '#999', borderBottom: '1px solid #303030', paddingBottom: 4 }}>
                  训练状态
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Tag color={loading ? 'processing' : training.historyData.length > 0 ? 'success' : 'default'}>
                    {loading ? '训练中...' : training.historyData.length > 0 ? '已完成' : '待开始'}
                  </Tag>
                  {loading && <span style={{ fontSize: 11, color: '#999' }}>请勿关闭页面</span>}
            </div>
          </Card>
            </Col>
          </Row>
        </Col>
      </Row>
    </div>
  );
};

export default TrainingPage;
