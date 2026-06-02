import React from 'react';
import {Composition} from 'remotion';
import {VantageExplainer} from './VantageExplainer';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="VantageExplainer"
      component={VantageExplainer}
      durationInFrames={540}
      fps={12}
      width={960}
      height={540}
    />
  );
};
