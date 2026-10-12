import React, { useState } from 'react';
import { 
  Card, Row, Col, Input, Button, Radio, InputNumber, Form, 
  Table, Progress, Tag, Image, Space, Divider, message
} from 'antd';
import { 
  FolderOpenOutlined, 
  PlayCircleOutlined, 
  PauseCircleOutlined,
  SwapRightOutlined,
  CopyOutlined,
  SettingOutlined,
  LeftOutlined,
  RightOutlined
} from '@ant-design/icons';

// Mock 风格迁移示例数据
const STYLE_TRANSFER_SAMPLES = [
  { id: '000001', source: '/samples/kitti/000001.png', transferred: '/samples/k2c/000001.png' },
  { id: '000003', source: '/samples/kitti/000003.png', transferred: '/samples/k2c/000003.png' },
  { id: '000004', source: '/samples/kitti/000004.png', transferred: '/samples/k2c/000004.png' },
  { id: '000005', source: '/samples/kitti/000005.png', transferred: '/samples/k2c/000005.png' },
];

// Mock 任务队列数据
const MOCK_TASKS = [
  { key: '1', id: '#001', direction: 'S→T', progress: 100, status: 'completed', time: '05:23', output: 'output/K2C/', count: '7481/7481' },
  { key: '2', id: '#002', direction: 'T→S', progress: 0, status: 'queued', time: '-', output: 'output/C2K/', count: '0/2975' },
];

// Mock 日志
const MOCK_LOGS = [
  { type: 'info', message: '> [INFO] CUT model loaded successfully.' },
  { type: 'info', message: '> [INFO] Source dataset: 7481 images found.' },
  { type: 'info', message: '> [INFO] Processing batch 1/1871 (4 images)...' },
  { type: 'success', message: '> [SUCCESS] Task #001 completed in 05:23.' },
];

const StyleTransfer: React.FC = () => {
  const [form] = Form.useForm();
  const [running, setRunning] = useState(false);
  const [tasks] = useState(MOCK_TASKS);
  const [logs, setLogs] = useState(MOCK_LOGS);
  const [currentPage, setCurrentPage] = useState(1);

  const handleStart = () => {
    setRunning(true);
    message.loading({ content: '正在启动风格迁移任务...', key: 'transfer' });
    setTimeout(() => {
      message.success({ content: '任务已添加到队列', key: 'transfer' });
      setLogs(prev => [...prev, { type: 'info', message: `> [INFO] New task added at ${new Date().toLocaleTimeString()}` }]);
    }, 1000);
  };

  const handleStop = () => {
    setRunning(false);
    message.info('任务已停止');
  };

  const copyLogs = () => {
    navigator.clipboard.writeText(logs.map(l => l.message).join('\n'));
    message.success('日志已复制');
  };

  const taskColumns = [
    { title: 'ID', dataIndex: 'id', width: 50 },
    { title: '方向', dataIndex: 'direction', width: 50 },
    { title: '进度', dataIndex: 'progress', width: 100, render: (v: number) => <Progress percent={v} size="small" strokeColor={v === 100 ? '#52c41a' : '#1890ff'} /> },
    { title: '状态', dataIndex: 'status', width: 70, render: (s: string) => <Tag color={s === 'completed' ? 'success' : 'default'}>{s === 'completed' ? '完成' : '队列中'}</Tag> },
    { title: '已处理', dataIndex: 'count', width: 80 },
    { title: '输出', dataIndex: 'output', ellipsis: true },
  ];

  const pageSize = 2;
  const paginatedSamples = STYLE_TRANSFER_SAMPLES.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Row gutter={12} style={{ flex: 1, minHeight: 0 }}>
        {/* 左侧配置舱 */}
        <Col span={7}>
          <Card 
            title={<span style={{ fontSize: 13 }}><SettingOutlined /> 配置舱</span>}
            size="small"
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
            styles={{ body: { flex: 1, overflow: 'auto', padding: '8px 12px' }, header: { minHeight: 36, padding: '0 12px' } }}
          >
            <Form form={form} layout="vertical" size="small" initialValues={{
              sourcePath: 'dataset3/yolo_kitti_format_2/images/train',
              targetPath: 'dataset3/yolov5_city_format_2/images/train',
              direction: 's2t', resolution: '1024x512', batchSize: 4, numWorkers: 4,
              outputDir: 'output/K2C', naming: '{name}_transferred'
            }}>
              <Divider orientation={"left" as any} style={{ margin: '0 0 6px 0', fontSize: 11 }}>数据选择</Divider>
              <Form.Item label={<span style={{ fontSize: 11 }}>源域路径</span>} name="sourcePath" style={{ marginBottom: 4 }}>
                <Input prefix={<FolderOpenOutlined style={{ color: '#1890ff' }} />} style={{ fontSize: 11 }} />
              </Form.Item>
              <Form.Item label={<span style={{ fontSize: 11 }}>目标域路径</span>} name="targetPath" style={{ marginBottom: 4 }}>
                <Input prefix={<FolderOpenOutlined style={{ color: '#ff7875' }} />} style={{ fontSize: 11 }} />
              </Form.Item>

              <Divider orientation={"left" as any} style={{ margin: '6px 0', fontSize: 11 }}>迁移方向</Divider>
              <Form.Item name="direction" style={{ marginBottom: 4 }}>
                <Radio.Group size="small">
                  <Space direction="vertical" size={0}>
                    <Radio value="s2t" style={{ fontSize: 11 }}><Tag color="blue" style={{ fontSize: 10 }}>S→T</Tag> 源→目标</Radio>
                    <Radio value="t2s" style={{ fontSize: 11 }}><Tag color="red" style={{ fontSize: 10 }}>T→S</Tag> 目标→源</Radio>
                    <Radio value="both" style={{ fontSize: 11 }}><Tag color="purple" style={{ fontSize: 10 }}>双向</Tag></Radio>
                  </Space>
                </Radio.Group>
              </Form.Item>

              <Divider orientation={"left" as any} style={{ margin: '6px 0', fontSize: 11 }}>参数设置</Divider>
              <Row gutter={6}>
                <Col span={12}>
                  <Form.Item label={<span style={{ fontSize: 11 }}>分辨率</span>} name="resolution" style={{ marginBottom: 4 }}>
                    <Input style={{ fontSize: 11 }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label={<span style={{ fontSize: 11 }}>批大小</span>} name="batchSize" style={{ marginBottom: 4 }}>
                    <InputNumber min={1} max={32} style={{ width: '100%', fontSize: 11 }} />
                  </Form.Item>
                </Col>
              </Row>

              <Divider orientation={"left" as any} style={{ margin: '6px 0', fontSize: 11 }}>输出设置</Divider>
              <Form.Item label={<span style={{ fontSize: 11 }}>输出目录</span>} name="outputDir" style={{ marginBottom: 4 }}>
                <Input prefix={<FolderOpenOutlined />} style={{ fontSize: 11 }} />
              </Form.Item>
            </Form>

            <div style={{ marginTop: 'auto', paddingTop: 6 }}>
              {!running ? (
                <Button type="primary" icon={<PlayCircleOutlined />} block size="small" onClick={handleStart}>开始迁移</Button>
              ) : (
                <Button danger icon={<PauseCircleOutlined />} block size="small" onClick={handleStop}>停止任务</Button>
              )}
            </div>
          </Card>
        </Col>

        {/* 右侧：预览区 + 任务队列 */}
        <Col span={17} style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* 结果预览 */}
          <Card 
            title={<span style={{ fontSize: 13 }}>结果预览</span>}
            size="small"
            style={{ flex: 1, display: 'flex', flexDirection: 'column', marginBottom: 8 }}
            styles={{ body: { flex: 1, overflow: 'auto', padding: '8px' }, header: { minHeight: 36, padding: '0 12px' } }}
            extra={
              <Space size={4}>
                <Button size="small" icon={<LeftOutlined />} disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)} />
                <span style={{ fontSize: 11 }}>{currentPage}</span>
                <Button size="small" icon={<RightOutlined />} disabled={currentPage >= Math.ceil(STYLE_TRANSFER_SAMPLES.length / pageSize)} onClick={() => setCurrentPage(p => p + 1)} />
              </Space>
            }
          >
            <Row gutter={[8, 8]}>
              {paginatedSamples.map(sample => (
                <Col span={12} key={sample.id}>
                  <Card size="small" style={{ background: '#141414' }} styles={{ body: { padding: 8 } }}>
                    <Row gutter={6} align="middle">
                      <Col span={11}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="blue" style={{ marginBottom: 2, fontSize: 10 }}>源域</Tag>
                          <Image src={sample.source} style={{ width: '100%', height: 80, objectFit: 'cover', borderRadius: 4 }} preview={{ mask: '查看' }} />
                          <div style={{ fontSize: 9, color: '#666', marginTop: 2 }}>{sample.id}.png</div>
                        </div>
                      </Col>
                      <Col span={2} style={{ textAlign: 'center' }}>
                        <SwapRightOutlined style={{ fontSize: 16, color: '#1890ff' }} />
                      </Col>
                      <Col span={11}>
                        <div style={{ textAlign: 'center' }}>
                          <Tag color="green" style={{ marginBottom: 2, fontSize: 10 }}>迁移后</Tag>
                          <Image src={sample.transferred} style={{ width: '100%', height: 80, objectFit: 'cover', borderRadius: 4 }} preview={{ mask: '查看' }} />
                          <div style={{ fontSize: 9, color: '#666', marginTop: 2 }}>{sample.id}_t.png</div>
                        </div>
                      </Col>
                    </Row>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>

          {/* 任务队列与日志 */}
          <Card size="small" style={{ flexShrink: 0 }} styles={{ body: { padding: '6px 10px' } }}>
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

export default StyleTransfer;
