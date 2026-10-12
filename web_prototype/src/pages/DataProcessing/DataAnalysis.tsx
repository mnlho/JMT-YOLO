import React, { useState } from 'react';
import { Card, Row, Col, Input, Button, Statistic, Tag, Image, Space, Divider, message, Spin, Typography } from 'antd';
import { FolderOpenOutlined, SearchOutlined, DatabaseOutlined, PictureOutlined, BarChartOutlined, SwapOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';

const { Text, Title } = Typography;

// Mock 数据
const MOCK_ANALYSIS_RESULT = {
  source: {
    name: 'Cityscapes', imageCount: 2975, totalObjects: 50588,
    classes: ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'],
    classDistribution: [
      { name: 'car', count: 26963 }, { name: 'person', count: 17900 }, { name: 'bicycle', count: 3658 },
      { name: 'rider', count: 1755 }, { name: 'motorcycle', count: 734 }, { name: 'bus', count: 386 },
      { name: 'truck', count: 482 }, { name: 'train', count: 168 }
    ]
  },
  target: {
    name: 'Foggy Cityscapes', imageCount: 2975, totalObjects: 50588,
    classes: ['bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'],
    classDistribution: [
      { name: 'car', count: 26963 }, { name: 'person', count: 17900 }, { name: 'bicycle', count: 3658 },
      { name: 'rider', count: 1755 }, { name: 'motorcycle', count: 734 }, { name: 'bus', count: 386 },
      { name: 'truck', count: 482 }, { name: 'train', count: 168 }
    ]
  },
  domainGap: { featureDistance: 0.342, brightnessGap: 0.27, contrastGap: 0.31, clarityGap: 0.53 }
};

const SOURCE_SAMPLE_IMAGES = ['/samples/source_aachen_000000_000019.jpg', '/samples/source_aachen_000001_000019.jpg', '/samples/source_aachen_000002_000019.jpg', '/samples/source_aachen_000003_000019.jpg'];
const TARGET_SAMPLE_IMAGES = ['/samples/target_aachen_000000_000019.jpg', '/samples/target_aachen_000001_000019.jpg', '/samples/target_aachen_000002_000019.jpg', '/samples/target_aachen_000003_000019.jpg'];

const DataAnalysis: React.FC = () => {
  const [sourcePath, setSourcePath] = useState('/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainvaltest/yolov5_format');
  const [targetPath, setTargetPath] = useState('/home/yu/zjh/SSDA-YOLO-copy/dataset/leftImg8bit_trainval_foggyDBF/yolov5_format');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<typeof MOCK_ANALYSIS_RESULT | null>(null);

  const handleAnalyze = async () => {
    if (!sourcePath || !targetPath) { message.warning('请输入数据集路径'); return; }
    setAnalyzing(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setAnalysisResult(MOCK_ANALYSIS_RESULT);
    setAnalyzing(false);
    message.success('数据分析完成');
  };

  const getClassDistributionOption = () => {
    if (!analysisResult) return {};
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { data: ['源域', '目标域'], top: 5, right: 10, textStyle: { fontSize: 10 }, orient: 'horizontal', itemWidth: 12, itemHeight: 8 },
      grid: { left: 45, right: 15, top: 30, bottom: 25 },
      xAxis: { type: 'category', data: analysisResult.source.classDistribution.map(c => c.name), axisLabel: { rotate: 30, fontSize: 9, interval: 0 } },
      yAxis: { type: 'value', axisLabel: { fontSize: 9 } },
      series: [
        { name: '源域', type: 'bar', data: analysisResult.source.classDistribution.map(c => c.count), itemStyle: { color: '#1890ff' }, barWidth: '35%' },
        { name: '目标域', type: 'bar', data: analysisResult.target.classDistribution.map(c => c.count), itemStyle: { color: '#ff7875' }, barWidth: '35%' }
      ]
    };
  };

  const getDomainGapRadarOption = () => {
    if (!analysisResult) return {};
    return {
      legend: { data: ['源域', '目标域'], top: 5, right: 10, textStyle: { fontSize: 9 }, orient: 'horizontal', itemWidth: 12, itemHeight: 8 },
      radar: { indicator: [{ name: '亮度', max: 1 }, { name: '对比度', max: 1 }, { name: '清晰度', max: 1 }, { name: '饱和度', max: 1 }, { name: '纹理', max: 1 }], radius: '55%', center: ['50%', '58%'], name: { textStyle: { fontSize: 9 } } },
      series: [{ type: 'radar', data: [
        { value: [0.85, 0.82, 0.91, 0.78, 0.72], name: '源域', areaStyle: { opacity: 0.3 }, lineStyle: { color: '#1890ff' }, itemStyle: { color: '#1890ff' } },
        { value: [0.58, 0.51, 0.38, 0.65, 0.68], name: '目标域', areaStyle: { opacity: 0.3 }, lineStyle: { color: '#ff7875' }, itemStyle: { color: '#ff7875' } }
      ]}]
    };
  };

  return (
    <div style={{ height: '100%', overflow: 'auto' }}>
      {/* 数据集导入区域 */}
      <Card size="small" title={<Space><DatabaseOutlined /><span style={{ fontSize: 13 }}>数据集配置</span></Space>} style={{ marginBottom: 8 }} styles={{ body: { padding: '8px 12px' }, header: { minHeight: 36, padding: '0 12px' } }}>
        <Row gutter={12} align="middle">
          <Col span={10}>
            <Space direction="vertical" style={{ width: '100%' }} size={2}>
              <Text type="secondary" style={{ fontSize: 11 }}><Tag color="blue" style={{ fontSize: 10 }}>源域</Tag> 路径</Text>
              <Input placeholder="源域数据集路径" prefix={<FolderOpenOutlined style={{ color: '#1890ff' }} />} value={sourcePath} onChange={e => setSourcePath(e.target.value)} style={{ fontSize: 11 }} />
            </Space>
          </Col>
          <Col span={10}>
            <Space direction="vertical" style={{ width: '100%' }} size={2}>
              <Text type="secondary" style={{ fontSize: 11 }}><Tag color="red" style={{ fontSize: 10 }}>目标域</Tag> 路径</Text>
              <Input placeholder="目标域数据集路径" prefix={<FolderOpenOutlined style={{ color: '#ff7875' }} />} value={targetPath} onChange={e => setTargetPath(e.target.value)} style={{ fontSize: 11 }} />
            </Space>
          </Col>
          <Col span={4}>
            <Button type="primary" icon={<SearchOutlined />} onClick={handleAnalyze} loading={analyzing} style={{ width: '100%', marginTop: 16 }} size="small">开始分析</Button>
          </Col>
        </Row>
      </Card>

      {analyzing && <div style={{ textAlign: 'center', padding: 40 }}><Spin size="large" /><div style={{ marginTop: 12, color: '#666', fontSize: 12 }}>正在分析数据集...</div></div>}

      {analysisResult && !analyzing && (
        <>
          {/* 统计卡片 */}
          <Row gutter={8} style={{ marginBottom: 8 }}>
            {[
              { title: '源域图像', value: analysisResult.source.imageCount, suffix: '张', bg: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' },
              { title: '目标域图像', value: analysisResult.target.imageCount, suffix: '张', bg: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' },
              { title: '类别数量', value: analysisResult.source.classes.length, suffix: '类', bg: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)' },
              { title: '标注目标', value: analysisResult.source.totalObjects, suffix: '个', bg: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)' },
            ].map((item, idx) => (
              <Col span={6} key={idx}>
                <Card size="small" style={{ background: item.bg }} styles={{ body: { padding: '8px 12px' } }}>
                  <Statistic title={<span style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11 }}>{item.title}</span>} value={item.value} valueStyle={{ color: '#fff', fontSize: 18 }} suffix={<span style={{ fontSize: 10, color: 'rgba(255,255,255,0.7)' }}>{item.suffix}</span>} />
                </Card>
              </Col>
            ))}
          </Row>

          {/* 类别标签 */}
          <Card size="small" style={{ marginBottom: 8 }} styles={{ body: { padding: '6px 12px' } }}>
            <Space wrap size={4}>
              <Text strong style={{ fontSize: 11 }}>类别:</Text>
              {analysisResult.source.classes.map((cls, idx) => <Tag key={cls} color={['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'gold'][idx % 8]} style={{ fontSize: 10 }}>{cls}</Tag>)}
            </Space>
          </Card>

          {/* 图表 + 示例图像 */}
          <Row gutter={8} style={{ marginBottom: 8 }}>
            <Col span={14}>
              <Card size="small" title={<Space><BarChartOutlined /><span style={{ fontSize: 12 }}>类别分布对比</span></Space>} styles={{ body: { padding: 8 }, header: { minHeight: 32, padding: '0 12px' } }}>
                <ReactECharts option={getClassDistributionOption()} style={{ height: 160 }} />
              </Card>
            </Col>
            <Col span={10}>
              <Card size="small" title={<Space><SwapOutlined /><span style={{ fontSize: 12 }}>域差异分析</span></Space>} styles={{ body: { padding: 8 }, header: { minHeight: 32, padding: '0 12px' } }}>
                <ReactECharts option={getDomainGapRadarOption()} style={{ height: 160 }} />
              </Card>
            </Col>
          </Row>

          {/* 示例图像 */}
          <Row gutter={8}>
            <Col span={12}>
              <Card size="small" title={<Space><PictureOutlined style={{ color: '#1890ff' }} /><span style={{ fontSize: 12 }}>源域示例</span><Tag color="blue" style={{ fontSize: 10 }}>Cityscapes</Tag></Space>} styles={{ body: { padding: 8 }, header: { minHeight: 32, padding: '0 12px' } }}>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <Image.PreviewGroup>
                    <Space size={6}>
                      {SOURCE_SAMPLE_IMAGES.map((src, idx) => <Image key={idx} src={src} style={{ width: 100, height: 50, objectFit: 'cover', borderRadius: 4, border: '2px solid #1890ff' }} preview={{ mask: <span style={{ fontSize: 9 }}>查看</span> }} />)}
                    </Space>
                  </Image.PreviewGroup>
                </div>
                <div style={{ textAlign: 'center', marginTop: 4 }}><Text type="secondary" style={{ fontSize: 10 }}>晴朗天气 · 高清晰度</Text></div>
              </Card>
            </Col>
            <Col span={12}>
              <Card size="small" title={<Space><PictureOutlined style={{ color: '#ff7875' }} /><span style={{ fontSize: 12 }}>目标域示例</span><Tag color="red" style={{ fontSize: 10 }}>Foggy</Tag></Space>} styles={{ body: { padding: 8 }, header: { minHeight: 32, padding: '0 12px' } }}>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <Image.PreviewGroup>
                    <Space size={6}>
                      {TARGET_SAMPLE_IMAGES.map((src, idx) => <Image key={idx} src={src} style={{ width: 100, height: 50, objectFit: 'cover', borderRadius: 4, border: '2px solid #ff7875' }} preview={{ mask: <span style={{ fontSize: 9 }}>查看</span> }} />)}
                    </Space>
                  </Image.PreviewGroup>
                </div>
                <div style={{ textAlign: 'center', marginTop: 4 }}><Text type="secondary" style={{ fontSize: 10 }}>雾天场景 · 低清晰度</Text></div>
              </Card>
            </Col>
          </Row>
        </>
      )}

      {!analysisResult && !analyzing && (
        <div style={{ textAlign: 'center', padding: 60, color: '#999' }}>
          <DatabaseOutlined style={{ fontSize: 40, marginBottom: 12 }} />
          <div style={{ fontSize: 12 }}>请输入数据集路径并点击"开始分析"</div>
        </div>
      )}
    </div>
  );
};

export default DataAnalysis;
