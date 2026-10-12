import React from 'react';
import { Tabs } from 'antd';
import { 
  SwapOutlined, 
  ScissorOutlined, 
  BarChartOutlined 
} from '@ant-design/icons';
import StyleTransfer from './StyleTransfer';
import SAMSegmentation from './SAMSegmentation';
import DataAnalysis from './DataAnalysis';

const DataProcessing: React.FC = () => {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Tabs 
        defaultActiveKey="style-transfer"
        type="card"
        style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
        items={[
          {
            key: 'style-transfer',
            label: (
              <span>
                <SwapOutlined />
                风格迁移
              </span>
            ),
            children: <StyleTransfer />
          },
          {
            key: 'sam-segment',
            label: (
              <span>
                <ScissorOutlined />
                SAM 分割
              </span>
            ),
            children: <SAMSegmentation />
          },
          {
            key: 'data-analysis',
            label: (
              <span>
                <BarChartOutlined />
                数据分析
              </span>
            ),
            children: <DataAnalysis />
          }
        ]}
      />
    </div>
  );
};

export default DataProcessing;
