import React, { useState } from 'react';
import { 
  Card, Row, Col, Input, Button, InputNumber, Form, Select,
  Table, Progress, Tag, Image, Space, Divider, message, Tabs, Statistic
} from 'antd';
import { 
  FolderOpenOutlined, 
  PlayCircleOutlined, 
  PauseCircleOutlined,
  CopyOutlined,
  SettingOutlined,
  ExportOutlined,
  LeftOutlined,
  RightOutlined
} from '@ant-design/icons';

// Mock SAM 分割示例数据
const SAM_SEGMENT_SAMPLES = [
  { id: '007470', original: '/samples/sam/007470.png', gtBoxes: '/samples/sam/007470_GT.png', mask: '/samples/sam/007470_clean.png', objects: 8, classes: { car: 8 } }
];

// Mock 任务队列数据
const MOCK_TASKS = [
  { key: '1', id: '#001', mode: 'Box Prompt', progress: 100, status: 'completed', processed: '100/100', output: 'seg_kitti/' },
];

// Mock 日志
const MOCK_LOGS = [
  { type: 'info', message: '> [INFO] Loading SAM vit_h model...' },
  { type: 'info', message: '> [INFO] Model loaded.' },
  { type: 'info', message: '> [INFO] Processing 007470.png with 8 box prompts...' },
  { type: 'success', message: '> [SUCCESS] Task #001 completed.' },
];

const SAMSegmentation: React.FC = () => {
  const [form] = Form.useForm();
  const [running, setRunning] = useState(false);
  const [tasks] = useState(MOCK_TASKS);
  const [logs, setLogs] = useState(MOCK_LOGS);
  const [promptMode, setPromptMode] = useState('box');
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);

  const currentSample = SAM_SEGMENT_SAMPLES[currentSampleIndex];

  const handleStart = () => {
    setRunning(true);
    message.loading({ content: '正在启动 SAM 分割任务...', key: 'sam' });
    setTimeout(() => {
      message.success({ content: 'SAM 分割任务已添加到队列', key: 'sam' });
      setLogs(prev => [...prev, { type: 'info', message: `> [INFO] New SAM task added at ${new Date().toLocaleTimeString()}` }]);
    }, 1000);
  };

  const handleStop = () => { setRunning(false); message.info('任务已停止'); };
  const handleExport = () => { message.success('分割标签已导出为 YOLO 格式'); };
  const copyLogs = () => { navigator.clipboard.writeText(logs.map(l => l.message).join('\n')); message.success('日志已复制'); };

  const taskColumns = [
    { title: 'ID', dataIndex: 'id', width: 50 },
    { title: 'Prompt', dataIndex: 'mode', width: 80 },
    { title: '进度', dataIndex: 'progress', width: 100, render: (v: number) => <Progress percent={v} size="small" strokeColor={v === 100 ? '#52c41a' : '#1890ff'} /> },
    { title: '状态', dataIndex: 'status', width: 60, render: (s: string) => <Tag color={s === 'completed' ? 'success' : 'default'}>{s === 'completed' ? '完成' : '队列'}</Tag> },
    { title: '已处理', dataIndex: 'processed', width: 80 },
    { title: '输出', dataIndex: 'output', ellipsis: true },
  ];

  const promptModeItems = [
    { key: 'box', label: 'Box', children: <div style={{ fontSize: 11, color: '#52c41a', padding: '4px 0' }}>✓ 推荐：从检测框直接生成分割掩码</div> },
    { key: 'text', label: 'Text', children: <Input.TextArea placeholder="car, vehicle..." rows={1} style={{ fontSize: 11 }} /> },
    { key: 'auto', label: 'Auto', children: <div style={{ fontSize: 11, color: '#999', padding: '4px 0' }}>YOLO检测 → SAM分割</div> }
  ];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Row gutter={12} style={{ flex: 1, minHeight: 0 }}>
        {/* 左侧工具箱 */}
        <Col span={7}>
          <Card 
            title={<span style={{ fontSize: 13 }}><SettingOutlined /> 工具箱</span>}
            size="small"
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            styles={{ body: { flex: 1, overflow: 'auto', padding: '8px 12px' }, header: { minHeight: 36, padding: '0 12px' } }}
          >
            <Form form={form} layout="vertical" size="small" initialValues={{
              datasetPath: 'dataset3/yolo_kitti_format_2/images/train',
              configPath: 'data/yamls_sda/KITTI_CityScapes_VOC.yaml',
              samModel: 'vit_h', maskThreshold: 0.5, outputFormat: 'png'
            }}>
              <Divider orientation={"left" as any} style={{ margin: '0 0 6px 0', fontSize: 11 }}>数据来源</Divider>
              <Form.Item label={<span style={{ fontSize: 11 }}>数据集路径</span>} name="datasetPath" style={{ marginBottom: 4 }}>
                <Input prefix={<FolderOpenOutlined style={{ color: '#1890ff' }} />} style={{ fontSize: 11 }} />
              </Form.Item>
              <Form.Item label={<span style={{ fontSize: 11 }}>yaml配置文件路径</span>} name="configPath" style={{ marginBottom: 4 }}>
                <Input prefix={<FolderOpenOutlined style={{ color: '#faad14' }} />} style={{ fontSize: 11 }} />
              </Form.Item>

              <Divider orientation={"left" as any} style={{ margin: '6px 0', fontSize: 11 }}>Prompt 模式</Divider>
              <Tabs activeKey={promptMode} onChange={setPromptMode} size="small" items={promptModeItems} style={{ marginBottom: 4 }} />

              <Divider orientation={"left" as any} style={{ margin: '6px 0', fontSize: 11 }}>推理设置</Divider>
              <Row gutter={6}>
                <Col span={12}>
                  <Form.Item label={<span style={{ fontSize: 11 }}>SAM模型</span>} name="samModel" style={{ marginBottom: 4 }}>
                    <Select size="small" options={[{ label: 'vit_h', value: 'vit_h' }, { label: 'vit_l', value: 'vit_l' }, { label: 'vit_b', value: 'vit_b' }]} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label={<span style={{ fontSize: 11 }}>阈值</span>} name="maskThreshold" style={{ marginBottom: 4 }}>
                    <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item label={<span style={{ fontSize: 11 }}>输出格式</span>} name="outputFormat" style={{ marginBottom: 4 }}>
                <Select size="small" options={[{ label: 'PNG', value: 'png' }, { label: 'COCO', value: 'coco' }, { label: 'YOLO', value: 'yolo' }]} />
              </Form.Item>
            </Form>

            <div style={{ marginTop: 'auto', paddingTop: 6 }}>
              <Row gutter={6}>
                <Col span={12}>
                  {!running ? (
                    <Button type="primary" icon={<PlayCircleOutlined />} block size="small" onClick={handleStart}>开始</Button>
                  ) : (
                    <Button danger icon={<PauseCircleOutlined />} block size="small" onClick={handleStop}>停止</Button>
                  )}
                </Col>
                <Col span={12}>
                  <Button icon={<ExportOutlined />} block size="small" onClick={handleExport}>导出</Button>
                </Col>
              </Row>
            </div>
          </Card>
        </Col>

        {/* 右侧：预览区 + 任务队列 */}
        <Col span={17} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* 结果预览 - 高度与左侧工具箱对齐 */}
          <Card 
            title={<span style={{ fontSize: 13 }}>结果预览</span>}
            size="small"
            style={{ height: 'calc(100% - 146px)', display: 'flex', flexDirection: 'column', marginBottom: 8 }}
            styles={{ body: { flex: 1, overflow: 'auto', padding: '8px' }, header: { minHeight: 36, padding: '0 12px' } }}
            extra={
              <Space size={4}>
                <Button size="small" icon={<LeftOutlined />} disabled={currentSampleIndex === 0} onClick={() => setCurrentSampleIndex(i => i - 1)} />
                <span style={{ fontSize: 11 }}>{currentSampleIndex + 1}</span>
                <Button size="small" icon={<RightOutlined />} disabled={currentSampleIndex >= SAM_SEGMENT_SAMPLES.length - 1} onClick={() => setCurrentSampleIndex(i => i + 1)} />
              </Space>
            }
          >
            {currentSample && (
              <>
                <Row gutter={8} style={{ marginBottom: 8 }}>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Tag color="blue" style={{ marginBottom: 2, fontSize: 10 }}>原始图像</Tag>
                      <Image src={currentSample.original} style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 4, border: '2px solid #1890ff' }} preview={{ mask: '查看' }} />
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Tag color="orange" style={{ marginBottom: 2, fontSize: 10 }}>检测框</Tag>
                      <Image src={currentSample.gtBoxes} style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 4, border: '2px solid #faad14' }} preview={{ mask: '查看' }} />
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Tag color="green" style={{ marginBottom: 2, fontSize: 10 }}>分割掩码</Tag>
                      <Image src={currentSample.mask} style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 4, border: '2px solid #52c41a' }} preview={{ mask: '查看' }} />
                    </div>
                  </Col>
                </Row>

                <Card size="small" style={{ background: 'linear-gradient(135deg, #1890ff20 0%, #52c41a20 100%)', border: '1px solid #1890ff40' }} styles={{ body: { padding: '8px 12px' } }}>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic title={<span style={{ fontSize: 10 }}>检测目标</span>} value={currentSample.objects} suffix="个" valueStyle={{ fontSize: 16, color: '#1890ff' }} />
                    </Col>
                    <Col span={8}>
                      <Statistic title={<span style={{ fontSize: 10 }}>分割掩码</span>} value={currentSample.objects} suffix="个" valueStyle={{ fontSize: 16, color: '#52c41a' }} />
                    </Col>
                    <Col span={8}>
                      <div style={{ fontSize: 10, color: '#666' }}>类别分布</div>
                      {Object.entries(currentSample.classes).map(([cls, count]) => (
                        <Tag key={cls} color="cyan" style={{ fontSize: 10, marginTop: 4 }}>{cls}: {count}</Tag>
                      ))}
                    </Col>
                  </Row>
                </Card>
              </>
            )}
          </Card>

          {/* 任务队列与日志 - 固定高度138px确保与左侧对齐 */}
          <Card size="small" style={{ height: 138, flexShrink: 0 }} styles={{ body: { padding: '6px 10px', height: 'calc(100% - 0px)', overflow: 'hidden' } }}>
            <Row gutter={12}>
              <Col span={14}>
                <div style={{ marginBottom: 2, fontWeight: 500, fontSize: 11 }}>任务队列</div>
                <Table columns={taskColumns} dataSource={tasks} size="small" pagination={false} scroll={{ y: 60 }} style={{ fontSize: 10 }} />
              </Col>
              <Col span={10}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                  <span style={{ fontWeight: 500, fontSize: 11 }}>日志</span>
                  <Button size="small" type="text" icon={<CopyOutlined />} onClick={copyLogs} style={{ fontSize: 10, height: 20 }}>复制</Button>
                </div>
                <div style={{ background: '#1e1e1e', borderRadius: 4, padding: 6, height: 76, overflow: 'auto', fontFamily: 'Consolas, Monaco, monospace', fontSize: 9 }}>
                  {logs.map((log, i) => (
                    <div key={i} style={{ color: log.type === 'success' ? '#52c41a' : '#d4d4d4', lineHeight: 1.4 }}>{log.message}</div>
                  ))}
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SAMSegmentation;
