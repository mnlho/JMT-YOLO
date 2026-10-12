import React, { useState } from 'react';
import { Layout, Menu, ConfigProvider } from 'antd';
import { 
  ExperimentOutlined, 
  CameraOutlined, 
  WifiOutlined,
  DatabaseOutlined
} from '@ant-design/icons';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { useAppStore } from '../store/appStore';

const { Header, Sider, Content } = Layout;

const MainLayout: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { isLiveMode, toggleMode } = useAppStore();

  const currentKey = location.pathname.replace('/', '') || 'training';

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh', height: '100vh' }}>
        {/* 左侧导航栏 */}
        <Sider 
          collapsible 
          collapsed={collapsed} 
          onCollapse={setCollapsed}
          theme="dark"
          style={{ 
            overflow: 'auto',
            height: '100vh',
            position: 'fixed',
            left: 0,
            top: 0,
            bottom: 0,
          }}
        >
          {/* Logo 区域 */}
          <div style={{ 
            height: 64, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            borderBottom: '1px solid rgba(255,255,255,0.1)'
          }}>
            <div style={{ 
              color: '#fff', 
              fontWeight: 700, 
              fontSize: collapsed ? 14 : 16,
              letterSpacing: 1
            }}>
              {collapsed ? '🎯' : '🎯 JMT-YOLO'}
            </div>
          </div>
          
          {/* 导航菜单 */}
          <Menu 
            theme="dark" 
            mode="inline" 
            selectedKeys={[currentKey]}
            style={{ borderRight: 0, marginTop: 8 }}
            items={[
              {
                key: 'data',
                icon: <DatabaseOutlined />,
                label: '数据处理',
                onClick: () => navigate('/data')
              },
              {
                key: 'training',
                icon: <ExperimentOutlined />,
                label: '模型训练',
                onClick: () => navigate('/training')
              },
              {
                key: 'inference',
                icon: <CameraOutlined />,
                label: '在线推理',
                onClick: () => navigate('/inference')
              }
            ]}
          />
          
          {/* 底部状态指示器 - 隐蔽的模式切换 (双击切换) */}
          <div style={{ 
            position: 'absolute', 
            bottom: 48, 
            width: '100%', 
            padding: '0 16px',
            textAlign: 'center'
          }}>
            <div 
              onDoubleClick={toggleMode}
                style={{ 
                color: 'rgba(255,255,255,0.25)',
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                gap: 8,
                padding: '4px 0',
                cursor: 'default',
                userSelect: 'none'
                }}
              >
                <WifiOutlined style={{ 
                color: isLiveMode ? '#52c41a' : 'rgba(255,255,255,0.15)',
                fontSize: 12
                }} />
              {!collapsed && <span style={{ fontSize: 11 }}>v1.0</span>}
            </div>
          </div>
        </Sider>
        
        {/* 右侧内容区 */}
        <Layout style={{ marginLeft: collapsed ? 80 : 200, transition: 'margin-left 0.2s' }}>
          {/* 顶部 Header */}
          <Header style={{ 
            padding: '0 24px', 
            background: '#fff', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            boxShadow: '0 1px 4px rgba(0,21,41,.08)',
            position: 'sticky',
            top: 0,
            zIndex: 10
          }}>
            <div style={{ fontSize: 18, fontWeight: 600, color: '#1f1f1f' }}>
              {currentKey === 'data' ? '数据处理' : currentKey === 'inference' ? '跨域目标检测演示' : '域自适应训练监控'}
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 12, color: '#999' }}>JMT-YOLO Demo System</span>
            </div>
          </Header>
          
          {/* 主内容区 */}
          <Content style={{ 
            margin: 16,
            padding: 24, 
            background: '#fff',
            borderRadius: 8,
            height: 'calc(100vh - 64px - 32px)',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
};

export default MainLayout;
