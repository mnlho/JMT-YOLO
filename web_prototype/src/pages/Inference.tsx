import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Select, Slider, InputNumber, Button, Upload, Tag, Statistic, Drawer, Empty, Image, Spin, Input, message, Tooltip, Descriptions } from 'antd';
import { InboxOutlined, RocketOutlined, DeleteOutlined, EyeOutlined, SettingOutlined } from '@ant-design/icons';
import { apiClient } from '../api/client';
import type { InferenceParams, DatasetInfo, ModelInfo } from '../api/client';
import { useAppStore } from '../store/appStore';

const { Dragger } = Upload;

interface DetectionResult {
  fileUrl: string;
  fileName: string;
  inferenceTime: number;
  imgSize: number;
  modelParams: string;
  objects: Array<{ class: string; class_id: number; conf: number; bbox: number[] }>;
  numObjects: number;
  resultImageBase64?: string;
}

const InferencePage: React.FC = () => {
  const isLive = useAppStore(state => state.isLiveMode);
  
  // 文件和结果状态
  const [fileList, setFileList] = useState<any[]>([]);
  const [results, setResults] = useState<DetectionResult[]>([]);
  const [processing, setProcessing] = useState(false);
  const [selectedResult, setSelectedResult] = useState<DetectionResult | null>(null);
  
  // 推理参数
  const [dataset, setDataset] = useState<'cityscapes' | 'kitti' | 'voc'>('cityscapes');
  const [weights, setWeights] = useState('/home/yu/zjh/SSDA-YOLO-copy/runs/train/12_21_bestpt_maskratio_03.6/weights/best_student.pt');
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [imgSize, setImgSize] = useState(960);
  
  // 可用模型和数据集列表
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);

  // 加载数据集和模型列表
  useEffect(() => {
    const loadData = async () => {
      const [datasetsData, modelsData] = await Promise.all([
        apiClient.getDatasets(),
        apiClient.getModels()
      ]);
      setDatasets(datasetsData);
      setModels(modelsData);
    };
    loadData();
  }, []);

  const handleUpload = ({ file, onSuccess }: any) => {
    setTimeout(() => {
      onSuccess?.("ok");
      const url = URL.createObjectURL(file);
      file.preview = url;
      setFileList(prev => [...prev, file]);
    }, 100);
  };

  const removeFile = (index: number) => {
    setFileList(prev => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFileList([]);
    setResults([]);
  };

  const runInference = async () => {
    if (fileList.length === 0) return;
    setProcessing(true);
    setResults([]);

    for (const file of fileList) {
      try {
        const params: InferenceParams = {
          file: file,
          weights: weights,
          dataset: dataset,
          confThres: confThreshold,
          imgSize: imgSize
        };
        
        const res = await apiClient.runInference(params);
        
        // 构建结果图片 URL
        let resultImageUrl = file.preview;
        if (res.resultImageBase64) {
          resultImageUrl = `data:image/jpeg;base64,${res.resultImageBase64}`;
        }
        
        setResults(prev => [...prev, {
          fileUrl: resultImageUrl,
          fileName: file.name,
          inferenceTime: res.inferenceTime,
          imgSize: imgSize,
          modelParams: res.modelParams,
          objects: res.objects,
          numObjects: res.numObjects,
          resultImageBase64: res.resultImageBase64
        }]);
      } catch (error: any) {
        console.error(error);
        message.error(`检测 ${file.name} 失败: ${error.message || '未知错误'}`);
      }
    }
    setProcessing(false);
    if (isLive) {
      message.success('检测完成！');
    }
  };

  // 获取类别对应的颜色
  const getClassColor = (className: string): string => {
    const colorMap: Record<string, string> = {
      'car': '#ff4d4f',
      'person': '#52c41a',
      'bicycle': '#1890ff',
      'motorcycle': '#faad14',
      'bus': '#722ed1',
      'truck': '#13c2c2',
      'train': '#eb2f96',
      'rider': '#fa541c',
      'aeroplane': '#2f54eb',
      'bird': '#a0d911',
      'boat': '#36cfc9',
      'bottle': '#ff85c0',
      'cat': '#ffc53d',
      'chair': '#597ef7',
      'cow': '#73d13d',
      'dog': '#ff7875',
      'horse': '#5cdbd3',
      'sheep': '#b37feb',
      'sofa': '#ffd666',
      'tvmonitor': '#69c0ff',
      'background': '#8c8c8c'
    };
    return colorMap[className.toLowerCase()] || '#1890ff';
  };

  const renderResultCard = (res: DetectionResult, index: number) => (
    <Card 
      key={index}
      hoverable
      size="small"
      style={{ marginBottom: 12 }}
      cover={
        <div style={{ position: 'relative', background: '#141414' }}>
          <img 
            alt={res.fileName} 
            src={res.fileUrl} 
            style={{ width: '100%', height: 180, objectFit: 'contain', display: 'block' }} 
          />
          <div style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            display: 'flex',
            gap: 4
          }}>
            <Button 
              type="primary" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={(e) => { e.stopPropagation(); setSelectedResult(res); }}
            />
          </div>
          {/* 显示检测数量标签 */}
          <div style={{
            position: 'absolute',
            top: 8,
            left: 8,
            background: 'rgba(0,0,0,0.7)',
            color: '#fff',
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 11
          }}>
            检测到 {res.numObjects} 个目标
          </div>
        </div>
      }
    >
      <Card.Meta 
        title={<span style={{ fontSize: 12 }}>{res.fileName}</span>}
        description={
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <Tag color="blue">{res.inferenceTime.toFixed(1)}ms</Tag>
            <Tag color="green">{res.imgSize}px</Tag>
            <Tag color="orange">{res.numObjects} 目标</Tag>
          </div>
        } 
      />
    </Card>
  );


  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 参数配置栏 */}
      <Card size="small" style={{ marginBottom: 12, flexShrink: 0 }}>
        <Row align="middle" gutter={12}>
          {/* 数据集选择 */}
          <Col flex="180px">
            <Tooltip title="选择模型对应的数据集类型">
              <Select 
                value={dataset}
                onChange={setDataset}
                style={{ width: '100%' }} 
                options={datasets.map(d => ({ 
                  label: d.name, 
                  value: d.id 
                }))}
                placeholder="选择数据集"
              />
            </Tooltip>
          </Col>
          
          {/* 模型选择 */}
          <Col flex="280px">
            <Tooltip title="输入模型权重路径或从列表选择">
              <Select
                value={weights}
                onChange={setWeights}
                style={{ width: '100%' }}
                showSearch
                allowClear
                placeholder="选择或输入模型路径"
                dropdownRender={(menu) => (
                  <div>
                    {menu}
                    <div style={{ padding: '8px', borderTop: '1px solid #303030' }}>
                      <Input
                        placeholder="自定义路径..."
                        value={weights}
                        onChange={(e) => setWeights(e.target.value)}
                        size="small"
                      />
                    </div>
                  </div>
                )}
                options={models.map(m => ({
                  label: <span style={{ fontSize: 11 }}>{m.name} ({m.size.toFixed(1)}MB)</span>,
                  value: m.path
                }))}
              />
            </Tooltip>
          </Col>
          
          {/* 图片尺寸 */}
          <Col flex="100px">
            <InputNumber 
              addonBefore={<span style={{ fontSize: 11 }}>Size</span>}
              value={imgSize}
              onChange={(v) => setImgSize(v || 640)}
              min={320}
              max={1280}
              step={32}
              style={{ width: '100%' }} 
            />
          </Col>
          
          {/* 置信度阈值 */}
          <Col flex="180px">
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ whiteSpace: 'nowrap', color: '#999', fontSize: 11 }}>置信度:</span>
              <Slider 
                value={confThreshold}
                onChange={setConfThreshold}
                min={0.1} 
                max={0.9} 
                step={0.05} 
                style={{ flex: 1, margin: 0 }} 
                tooltip={{ formatter: (v) => v?.toFixed(2) }}
              />
              <span style={{ minWidth: 32, fontSize: 12 }}>{confThreshold.toFixed(2)}</span>
            </div>
          </Col>
          
          {/* 操作按钮 */}
          <Col flex="auto" style={{ textAlign: 'right' }}>
            <Button onClick={clearAll} icon={<DeleteOutlined />} style={{ marginRight: 8 }}>清空</Button>
            <Button 
              type="primary" 
              icon={<RocketOutlined />} 
              loading={processing}
              onClick={runInference}
              disabled={fileList.length === 0}
            >
              开始检测 ({fileList.length})
            </Button>
          </Col>
        </Row>
        
      </Card>

      {/* 主内容区 */}
      <Row gutter={12} style={{ flex: 1, minHeight: 0 }}>
        {/* 左侧上传区 */}
        <Col span={7} style={{ height: '100%' }}>
          <Card 
            title={<><InboxOutlined /> 上传图片</>} 
            size="small" 
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' } }}
          >
            <Dragger 
              multiple 
              customRequest={handleUpload} 
              showUploadList={false}
              accept="image/*"
              style={{ marginBottom: 12 }}
            >
              <p style={{ marginBottom: 4 }}>
                <InboxOutlined style={{ fontSize: 28, color: '#1890ff' }} />
              </p>
              <p style={{ margin: 0, fontSize: 12 }}>点击或拖拽上传图片</p>
              <p style={{ margin: 0, fontSize: 11, color: '#666' }}>支持 Foggy Cityscapes / Clipart 等跨域图像</p>
            </Dragger>
            
            <div style={{ 
              flex: 1,
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: 6,
              overflowY: 'auto',
              alignContent: 'start'
            }}>
              {fileList.map((file, i) => (
                <div key={i} style={{ position: 'relative' }}>
                  <Image
                    src={file.preview}
                    style={{ 
                      width: '100%', 
                      height: 55, 
                      objectFit: 'cover', 
                      borderRadius: 4,
                      border: '1px solid #303030'
                    }}
                    preview={false}
                  />
                  <Button
                    type="text"
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => removeFile(i)}
                    style={{ 
                      position: 'absolute', 
                      top: 2, 
                      right: 2,
                      background: 'rgba(0,0,0,0.6)',
                      padding: '0 4px',
                      height: 18,
                      minWidth: 18
                    }}
                  />
                </div>
              ))}
            </div>
          </Card>
        </Col>

        {/* 右侧检测结果 */}
        <Col span={17} style={{ height: '100%' }}>
          <Card 
            title={<><SettingOutlined /> 检测结果</>}
            size="small" 
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            styles={{ body: { flex: 1, overflow: 'auto' } }}
          >
            {processing ? (
              <div style={{ textAlign: 'center', padding: 60 }}>
                <Spin size="large" />
                <p style={{ marginTop: 16, color: '#999' }}>
                  正在进行目标检测...
                  <br />
                  <span style={{ fontSize: 12 }}>使用 {dataset.toUpperCase()} 数据集配置</span>
                </p>
              </div>
            ) : results.length > 0 ? (
              <Row gutter={10}>
                {results.map((res, i) => (
                  <Col span={8} key={i}>
                    {renderResultCard(res, i)}
                  </Col>
                ))}
              </Row>
            ) : (
              <Empty 
                description={
                  <div>
                    <p style={{ margin: 0 }}>上传图片并点击开始检测</p>
                    <p style={{ margin: 0, fontSize: 12, color: '#666' }}>
                      支持 Cityscapes / KITTI / VOC 等跨域场景
                    </p>
                  </div>
                }
                style={{ padding: 60 }}
              />
            )}
          </Card>
        </Col>
      </Row>
      
      {/* 检测详情抽屉 */}
      <Drawer
        title="检测详情"
        placement="right"
        onClose={() => setSelectedResult(null)}
        open={!!selectedResult}
        width={480}
      >
        {selectedResult && (
          <div>
            {/* 检测结果图片 */}
            <div style={{ 
              background: '#141414', 
              borderRadius: 8, 
              marginBottom: 16,
              padding: 8
            }}>
              <Image 
                src={selectedResult.fileUrl} 
                style={{ width: '100%', borderRadius: 4 }} 
              />
            </div>
            
            {/* 性能指标 */}
            <Card size="small" title="性能指标" style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic 
                    title={<span style={{ fontSize: 11 }}>推理耗时</span>}
                    value={selectedResult.inferenceTime.toFixed(1)} 
                    suffix="ms" 
                    valueStyle={{ color: '#52c41a', fontSize: 18 }} 
                  />
                </Col>
                <Col span={8}>
                  <Statistic 
                    title={<span style={{ fontSize: 11 }}>图片尺寸</span>}
                    value={selectedResult.imgSize} 
                    suffix="px" 
                    valueStyle={{ fontSize: 18 }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic 
                    title={<span style={{ fontSize: 11 }}>模型参数</span>}
                    value={selectedResult.modelParams}
                    valueStyle={{ fontSize: 18 }}
                  />
                </Col>
              </Row>
            </Card>

            {/* 检测目标列表 */}
            <Card size="small" title={`检测目标 (${selectedResult.numObjects})`}>
              {selectedResult.objects.length > 0 ? (
                <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                  {selectedResult.objects.map((obj, i) => (
                    <div 
                      key={i} 
                      style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        padding: '8px 0',
                        borderBottom: i < selectedResult.objects.length - 1 ? '1px solid #303030' : 'none'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Tag color={getClassColor(obj.class)}>{obj.class}</Tag>
                        <span style={{ fontSize: 11, color: '#666' }}>ID: {obj.class_id}</span>
                      </div>
                      <span style={{ color: '#52c41a', fontWeight: 500 }}>
                        {(obj.conf * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <Empty description="未检测到目标" />
              )}
            </Card>

            {/* 边界框信息 */}
            {selectedResult.objects.length > 0 && (
              <Card size="small" title="边界框坐标" style={{ marginTop: 16 }}>
                <Descriptions size="small" column={2} labelStyle={{ fontSize: 11 }} contentStyle={{ fontSize: 11 }}>
                  {selectedResult.objects.slice(0, 5).map((obj, i) => (
                    <Descriptions.Item key={i} label={obj.class}>
                      [{obj.bbox.map(v => v.toFixed(0)).join(', ')}]
                    </Descriptions.Item>
                  ))}
                </Descriptions>
                {selectedResult.objects.length > 5 && (
                  <div style={{ textAlign: 'center', color: '#666', fontSize: 11, marginTop: 8 }}>
                    还有 {selectedResult.objects.length - 5} 个目标...
                  </div>
                )}
              </Card>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default InferencePage;
