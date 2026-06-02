import React from 'react';
import {
  AbsoluteFill,
  Easing,
  interpolate,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

const colors = {
  base: '#0a0a0a',
  level1: '#0f0f0f',
  level2: '#111111',
  level3: '#141414',
  border: '#222222',
  borderStrong: '#333333',
  text: '#ffffff',
  secondary: '#a1a1a1',
  muted: '#666666',
  accent: '#00e4b4',
  warning: '#f59e0b',
  error: '#ef4444',
};

const font =
  'Geist, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const mono =
  '"Geist Mono", "SF Mono", Consolas, "Liberation Mono", Menlo, monospace';

const ease = Easing.bezier(0.16, 1, 0.3, 1);

const clamp = {
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
  easing: ease,
};

const sec = (seconds: number, fps: number) => seconds * fps;

const fadeUp = (frame: number, start: number, duration: number) => ({
  opacity: interpolate(frame, [start, start + duration], [0, 1], clamp),
  transform: `translateY(${interpolate(
    frame,
    [start, start + duration],
    [18, 0],
    clamp
  )}px)`,
});

const Card: React.FC<{
  children: React.ReactNode;
  x: number;
  y: number;
  width: number;
  height: number;
  style?: React.CSSProperties;
}> = ({children, x, y, width, height, style}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width,
      height,
      background: colors.level2,
      border: `1px solid ${colors.border}`,
      borderRadius: 0,
      padding: 18,
      boxSizing: 'border-box',
      overflow: 'hidden',
      ...style,
    }}
  >
    {children}
  </div>
);

const Badge: React.FC<{
  children: React.ReactNode;
  tone?: 'neutral' | 'accent' | 'warning' | 'danger';
}> = ({children, tone = 'neutral'}) => {
  const toneColor =
    tone === 'accent'
      ? colors.accent
      : tone === 'warning'
        ? colors.warning
        : tone === 'danger'
          ? colors.error
          : colors.secondary;
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        height: 22,
        padding: '0 8px',
        background: '#1a1a1a',
        border: `1px solid ${
          tone === 'neutral' ? colors.borderStrong : `${toneColor}55`
        }`,
        borderRadius: 4,
        color: toneColor,
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: 0,
        textTransform: 'uppercase',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </div>
  );
};

const Label: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div
    style={{
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: 0,
      textTransform: 'uppercase',
      color: colors.muted,
      marginBottom: 10,
    }}
  >
    {children}
  </div>
);

const CodeLine: React.FC<{
  children: React.ReactNode;
  active?: boolean;
  dim?: boolean;
}> = ({children, active, dim}) => (
  <div
    style={{
      height: 25,
      display: 'flex',
      alignItems: 'center',
      padding: '0 8px',
      marginBottom: 3,
      borderRadius: 4,
      color: dim ? colors.muted : active ? colors.text : colors.secondary,
      background: active ? '#00e4b414' : 'transparent',
      border: active ? `1px solid ${colors.accent}44` : '1px solid transparent',
      fontFamily: mono,
      fontSize: 13,
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </div>
);

const Token: React.FC<{
  text: string;
  x: number;
  y: number;
  frame: number;
  start: number;
  tone?: 'accent' | 'muted' | 'danger' | 'warning';
  width?: number;
}> = ({text, x, y, frame, start, tone = 'accent', width}) => {
  const opacity = interpolate(frame, [start, start + 8], [0, 1], clamp);
  const tx = interpolate(frame, [start, start + 16], [-28, 0], clamp);
  const color =
    tone === 'accent'
      ? colors.accent
      : tone === 'danger'
        ? colors.error
        : tone === 'warning'
          ? colors.warning
          : colors.secondary;
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width,
        opacity,
        transform: `translateX(${tx}px)`,
        fontFamily: mono,
        fontSize: 12,
        color,
        border: `1px solid ${color}55`,
        background: `${color}14`,
        borderRadius: 4,
        padding: '5px 8px',
        boxSizing: 'border-box',
        whiteSpace: 'nowrap',
      }}
    >
      {text}
    </div>
  );
};

const Arrow: React.FC<{
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  progress: number;
  color?: string;
  dashed?: boolean;
}> = ({x1, y1, x2, y2, progress, color = colors.accent, dashed}) => {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const endX = x1 + dx * progress;
  const endY = y1 + dy * progress;
  const angle = Math.atan2(dy, dx);
  const head = progress > 0.96 ? 1 : 0;
  return (
    <svg
      width={960}
      height={540}
      style={{position: 'absolute', inset: 0, overflow: 'visible'}}
    >
      <line
        x1={x1}
        y1={y1}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={2}
        strokeDasharray={dashed ? '6 6' : undefined}
        opacity={0.85}
      />
      <path
        d={`M ${endX} ${endY} l ${-10 * Math.cos(angle - 0.5)} ${
          -10 * Math.sin(angle - 0.5)
        } M ${endX} ${endY} l ${-10 * Math.cos(angle + 0.5)} ${
          -10 * Math.sin(angle + 0.5)
        }`}
        stroke={color}
        strokeWidth={2}
        opacity={head}
        fill="none"
      />
    </svg>
  );
};

const PhaseHeader: React.FC<{
  frame: number;
  phase: string;
  title: string;
  subtitle: string;
}> = ({frame, phase, title, subtitle}) => {
  const intro = fadeUp(frame, 0, 12);
  return (
    <div
      style={{
        position: 'absolute',
        left: 48,
        top: 34,
        right: 48,
        display: 'flex',
        justifyContent: 'space-between',
        gap: 24,
        ...intro,
      }}
    >
      <div>
        <div
          style={{
            color: colors.muted,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0,
            textTransform: 'uppercase',
            marginBottom: 8,
          }}
        >
          {phase}
        </div>
        <div
          style={{
            color: colors.text,
            fontSize: 32,
            lineHeight: 1.08,
            fontWeight: 300,
            letterSpacing: 0,
          }}
        >
          {title}
        </div>
      </div>
      <div
        style={{
          color: colors.secondary,
          fontSize: 14,
          lineHeight: 1.45,
          width: 352,
          paddingTop: 22,
        }}
      >
        {subtitle}
      </div>
    </div>
  );
};

const BaselineScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const arrowProgress = interpolate(frame, [sec(1.6, fps), sec(3.0, fps)], [0, 1], clamp);
  const copied = interpolate(frame, [sec(3.2, fps), sec(4.5, fps)], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="Step 1 / ordinary PLD"
        title="Prompt Lookup Decoding copies text from the prompt"
        subtitle="For many code edits, the next output repeats large parts of the reference. PLD drafts those repeated tokens before the target model checks them."
      />
      <Card x={48} y={158} width={402} height={314} style={fadeUp(frame, 12, 12)}>
        <Label>Visible prompt reference</Label>
        <CodeLine>def get_user_name(user):</CodeLine>
        <CodeLine active>    return user.name</CodeLine>
        <CodeLine>    # copied structure</CodeLine>
        <div style={{position: 'absolute', left: 18, bottom: 18}}>
          <Badge tone="neutral">literal lookup source</Badge>
        </div>
      </Card>
      <Card x={560} y={158} width={352} height={314} style={fadeUp(frame, 22, 12)}>
        <Label>Drafted output</Label>
        <CodeLine>def get_user_name(user):</CodeLine>
        <CodeLine active>    return user.name</CodeLine>
        <div
          style={{
            marginTop: 24,
            color: colors.secondary,
            fontSize: 13,
            lineHeight: 1.45,
          }}
        >
          When the output matches the prompt text, PLD can propose a long draft
          and the target verifies many tokens in one pass.
        </div>
      </Card>
      <Arrow x1={450} y1={265} x2={560} y2={265} progress={arrowProgress} />
      <Token text="draft copy" x={462} y={288} frame={frame} start={36} width={92} />
      <div
        style={{
          position: 'absolute',
          left: 580,
          top: 352,
          opacity: copied,
          color: colors.accent,
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: 0,
          textTransform: 'uppercase',
        }}
      >
        many copied tokens accepted
      </div>
    </AbsoluteFill>
  );
};

const DriftProblemScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const arrowProgress = interpolate(frame, [sec(1.8, fps), sec(3.4, fps)], [0, 1], clamp);
  const reject = interpolate(frame, [sec(3.8, fps), sec(5.0, fps)], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="Step 2 / the problem"
        title="A simple rename hides the copyable span"
        subtitle="The desired output has the same shape, but some words changed. Literal PLD sees the old words and proposes drafts the target must reject."
      />
      <Card x={48} y={158} width={402} height={314} style={fadeUp(frame, 12, 12)}>
        <Label>Reference in the prompt</Label>
        <CodeLine>def get_user_name(user):</CodeLine>
        <CodeLine active>    return user.name</CodeLine>
        <CodeLine dim># instruction: rename user to account</CodeLine>
        <div style={{position: 'absolute', left: 18, bottom: 18}}>
          <Badge tone="warning">literal PLD still sees user</Badge>
        </div>
      </Card>
      <Card x={560} y={158} width={352} height={314} style={fadeUp(frame, 22, 12)}>
        <Label>Target wants this continuation</Label>
        <CodeLine>def get_user_name(account):</CodeLine>
        <CodeLine active>    return account.name</CodeLine>
        <div style={{marginTop: 24, color: colors.secondary, fontSize: 13, lineHeight: 1.45}}>
          The target model wants <span style={{color: colors.accent}}>account.name</span>,
          not <span style={{color: colors.warning}}>user.name</span>.
          The code shape is still copyable, but literal lookup misses it.
        </div>
        <div
          style={{
            marginTop: 20,
            opacity: reject,
            color: colors.error,
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: 0,
            lineHeight: 1.35,
            textTransform: 'uppercase',
          }}
        >
          target rejects at the rename
        </div>
      </Card>
      <Arrow x1={450} y1={265} x2={560} y2={265} progress={arrowProgress} color={colors.warning} dashed />
      <Token text="bad draft" x={462} y={288} frame={frame} start={40} tone="warning" width={92} />
    </AbsoluteFill>
  );
};

const FixedPromptScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="Constraint"
        title="Why not just rewrite the visible prompt?"
        subtitle="Sometimes that is fine. But if the application needs the model's original prompt-conditioned greedy output, the visible prompt must stay fixed."
      />
      <Card x={72} y={166} width={372} height={286} style={fadeUp(frame, 12, 12)}>
        <Label>Visible prompt injection</Label>
        <div style={{fontSize: 26, fontWeight: 300, color: colors.text, lineHeight: 1.15}}>
          Change the prompt first
        </div>
        <div style={{marginTop: 22, color: colors.secondary, fontSize: 14, lineHeight: 1.5}}>
          This can make generation easier and faster. But it asks the model a
          different question, so the greedy output may change.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="warning">changed prompt</Badge>
        </div>
      </Card>
      <Card x={516} y={166} width={372} height={286} style={fadeUp(frame, 28, 12)}>
        <Label>VANTAGE setting</Label>
        <div style={{fontSize: 26, fontWeight: 300, color: colors.text, lineHeight: 1.15}}>
          Keep the prompt fixed
        </div>
        <div style={{marginTop: 22, color: colors.secondary, fontSize: 14, lineHeight: 1.5}}>
          VANTAGE changes only the draft source. The target model still sees
          the original prompt and decides the emitted output.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="accent">fixed prompt</Badge>
        </div>
      </Card>
    </AbsoluteFill>
  );
};

const RewriteViewScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const mapProgress = interpolate(frame, [sec(0.7, fps), sec(1.5, fps)], [0, 1], clamp);
  const hiddenProgress = interpolate(frame, [sec(2.0, fps), sec(3.0, fps)], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="VANTAGE"
        title="VANTAGE builds a hidden edited copy"
        subtitle="The prompt says what to rewrite. Rewrite-View Lookup applies that map internally and uses the result only for draft lookup."
      />
      <Card x={48} y={152} width={258} height={294} style={fadeUp(frame, 10, 12)}>
        <Label>Map found in the prompt</Label>
        <div style={{display: 'flex', alignItems: 'center', gap: 10}}>
          <Badge tone="warning">old</Badge>
          <div style={{fontFamily: mono, color: colors.text, fontSize: 18}}>user</div>
        </div>
        <div style={{height: 20}} />
        <div style={{display: 'flex', alignItems: 'center', gap: 10}}>
          <Badge tone="accent">new</Badge>
          <div style={{fontFamily: mono, color: colors.text, fontSize: 18}}>account</div>
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="neutral">read from prompt only</Badge>
        </div>
      </Card>
      <Card x={370} y={152} width={282} height={294} style={fadeUp(frame, 22, 12)}>
        <Label>Hidden rewrite view</Label>
        <CodeLine>def get_user_name(account):</CodeLine>
        <CodeLine active>    return account.name</CodeLine>
        <div
          style={{
            marginTop: 22,
            color: colors.secondary,
            fontSize: 13,
            lineHeight: 1.45,
          }}
        >
          This hidden copy now contains the text PLD needed but could not see:
          <span style={{color: colors.accent}}> account.name</span>.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="accent">hidden lookup view</Badge>
        </div>
      </Card>
      <Card x={716} y={152} width={196} height={294} style={fadeUp(frame, 34, 12)}>
        <Label>Visible prompt</Label>
        <div style={{fontSize: 32, color: colors.text, fontWeight: 300}}>unchanged</div>
        <div style={{marginTop: 18, color: colors.secondary, lineHeight: 1.45, fontSize: 13}}>
          The target model is still conditioned on the original prompt.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="neutral">fixed prompt</Badge>
        </div>
      </Card>
      <Arrow x1={306} y1={260} x2={370} y2={260} progress={mapProgress} />
      <Arrow x1={652} y1={260} x2={716} y2={260} progress={hiddenProgress} color={colors.secondary} dashed />
    </AbsoluteFill>
  );
};

const VerificationScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const draftProgress = interpolate(frame, [sec(0.8, fps), sec(1.6, fps)], [0, 1], clamp);
  const verifyProgress = interpolate(frame, [sec(2.0, fps), sec(3.1, fps)], [0, 1], clamp);
  const acceptGlow = interpolate(frame, [sec(3.2, fps), sec(4.1, fps)], [0, 1], clamp);
  const draftPanel = fadeUp(frame, 18, 14);
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="Verifier contract"
        title="The model still decides every token"
        subtitle="The hidden view only proposes. A draft token is emitted only if the target agrees; otherwise the target emits its own next token."
      />
      <Card x={48} y={154} width={256} height={302} style={fadeUp(frame, 10, 12)}>
        <Label>Suggestion from hidden view</Label>
        <div
          style={{
            marginTop: 48,
            border: `1px solid ${colors.borderStrong}`,
            background: colors.level3,
            padding: 14,
            ...draftPanel,
          }}
        >
          <div
            style={{
              color: colors.muted,
              fontSize: 10,
              fontWeight: 600,
              textTransform: 'uppercase',
              marginBottom: 12,
            }}
          >
            Draft proposal
          </div>
          <div style={{display: 'flex', gap: 6, alignItems: 'center'}}>
            {['account', '.name', '↵'].map((token) => (
              <div
                key={token}
                style={{
                  minWidth: token === '↵' ? 30 : undefined,
                  padding: token === '↵' ? '7px 9px' : '7px 10px',
                  border: `1px solid ${colors.accent}55`,
                  background: '#00e4b414',
                  color: colors.accent,
                  borderRadius: 4,
                  fontFamily: mono,
                  fontSize: 12,
                  fontWeight: 600,
                  textAlign: 'center',
                  boxSizing: 'border-box',
                }}
              >
                {token}
              </div>
            ))}
          </div>
        </div>
        <div
          style={{
            marginTop: 18,
            color: colors.secondary,
            fontSize: 12,
            lineHeight: 1.45,
            ...fadeUp(frame, 26, 12),
          }}
        >
          These tokens are proposed to the verifier. They are not emitted by the
          hidden view.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="accent">draft source only</Badge>
        </div>
      </Card>
      <Card x={362} y={154} width={246} height={302} style={fadeUp(frame, 22, 12)}>
        <Label>Target model check</Label>
        <div
          style={{
            border: `1px solid ${colors.borderStrong}`,
            background: colors.level3,
            height: 118,
            padding: 14,
            fontFamily: mono,
            color: colors.secondary,
            fontSize: 13,
            lineHeight: 1.65,
          }}
        >
          target says → account
          <br />
          target says → .name
          <br />
          mismatch → target token
        </div>
        <div style={{marginTop: 24, color: colors.secondary, fontSize: 13, lineHeight: 1.45}}>
          Rejected suggestions are discarded. They do not become part of the
          generated code.
        </div>
      </Card>
      <Card
        x={668}
        y={154}
        width={264}
        height={302}
        style={{
          ...fadeUp(frame, 34, 12),
          borderColor: acceptGlow > 0.5 ? '#00e4b488' : colors.border,
          boxShadow: `0 0 ${28 * acceptGlow}px rgba(0, 228, 180, 0.08)`,
        }}
      >
        <Label>Emitted prefix</Label>
        <CodeLine>def get_user_name(account):</CodeLine>
        <CodeLine active>    return account.name</CodeLine>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="accent">same greedy output</Badge>
        </div>
      </Card>
      <Arrow x1={304} y1={249} x2={362} y2={249} progress={draftProgress} />
      <Arrow x1={608} y1={249} x2={668} y2={249} progress={verifyProgress} />
    </AbsoluteFill>
  );
};

const SpeedScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const field = interpolate(frame, [sec(0.8, fps), sec(1.7, fps)], [0, 1.28], clamp);
  const style = interpolate(frame, [sec(1.1, fps), sec(2.0, fps)], [0, 1.64], clamp);
  const mixed = interpolate(frame, [sec(1.4, fps), sec(2.3, fps)], [0, 1.33], clamp);
  const step = interpolate(frame, [sec(1.2, fps), sec(2.2, fps)], [0, 1], clamp);
  const Bar: React.FC<{label: string; value: number; y: number; max: number}> = ({
    label,
    value,
    y,
    max,
  }) => (
    <div style={{position: 'absolute', left: 76, top: y, width: 438}}>
      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 8}}>
        <span style={{color: colors.secondary, fontSize: 13}}>{label}</span>
        <span style={{color: colors.text, fontFamily: mono, fontSize: 13}}>
          {value.toFixed(2)}x
        </span>
      </div>
      <div style={{height: 14, background: '#1a1a1a', border: `1px solid ${colors.border}`, borderRadius: 4}}>
        <div
          style={{
            height: '100%',
            width: `${Math.min(100, (value / max) * 100)}%`,
            background: colors.accent,
            borderRadius: 3,
          }}
        />
      </div>
    </div>
  );
  return (
    <AbsoluteFill>
      <PhaseHeader
        frame={frame}
        phase="Measured effect"
        title="Fewer verifier steps, faster decoding"
        subtitle="The speedup comes from accepting longer chunks per target check, not from changing the model's answer."
      />
      <Card x={48} y={152} width={564} height={330} style={fadeUp(frame, 12, 12)}>
        <Label>Measured speedup over tuned PLD</Label>
        <Bar label="Field substitutions" value={field} y={72} max={1.7} />
        <Bar label="Identifier-style substitutions" value={style} y={144} max={1.7} />
        <Bar label="Mixed suite" value={mixed} y={216} max={1.7} />
      </Card>
      <Card x={648} y={152} width={284} height={330} style={fadeUp(frame, 24, 12)}>
        <Label>Why it speeds up</Label>
        <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12}}>
          <div style={{background: colors.level3, border: `1px solid ${colors.border}`, padding: 14}}>
            <div style={{color: colors.muted, fontSize: 10, textTransform: 'uppercase', letterSpacing: 0}}>PLD</div>
            <div style={{color: colors.text, fontSize: 30, fontWeight: 300, marginTop: 10}}>
              {Math.round(interpolate(step, [0, 1], [752, 752]))}
            </div>
            <div style={{color: colors.secondary, fontSize: 12}}>verifier steps</div>
          </div>
          <div style={{background: '#00e4b410', border: `1px solid #00e4b455`, padding: 14}}>
            <div style={{color: colors.accent, fontSize: 10, textTransform: 'uppercase', letterSpacing: 0}}>VANTAGE</div>
            <div style={{color: colors.text, fontSize: 30, fontWeight: 300, marginTop: 10}}>
              {Math.round(interpolate(step, [0, 1], [752, 388]))}
            </div>
            <div style={{color: colors.secondary, fontSize: 12}}>field row</div>
          </div>
        </div>
        <div style={{marginTop: 24, color: colors.secondary, fontSize: 13, lineHeight: 1.5}}>
          Same fixed prompt. Same audited greedy output. Fewer expensive target
          checks on high-copy explicit-map rows.
        </div>
        <div style={{position: 'absolute', bottom: 18, left: 18}}>
          <Badge tone="neutral">controlled fp32/sdpa workloads</Badge>
        </div>
      </Card>
    </AbsoluteFill>
  );
};

const IntroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const line = interpolate(frame, [sec(1.0, fps), sec(2.0, fps)], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: 48, top: 78, ...fadeUp(frame, -10, 14)}}>
        <div
          style={{
            color: colors.muted,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0,
            textTransform: 'uppercase',
            marginBottom: 12,
          }}
        >
          Why code edits are hard to reuse
        </div>
        <div
          style={{
            fontSize: 48,
            lineHeight: 1.06,
            fontWeight: 300,
            letterSpacing: 0,
            color: colors.text,
            width: 740,
          }}
        >
          Code edits often copy most of the prompt
        </div>
        <div
          style={{
            marginTop: 24,
            width: 642,
            color: colors.secondary,
            fontSize: 16,
            lineHeight: 1.5,
          }}
        >
          Speculative decoding tries to guess the next tokens early. If the
          target model agrees with the guess, generation can move forward with
          fewer expensive target-model checks.
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: 48,
          right: 48,
          bottom: 92,
          height: 1,
          background: `linear-gradient(90deg, transparent, ${colors.borderStrong}, transparent)`,
          transform: `scaleX(${line})`,
          transformOrigin: 'left center',
        }}
      />
      <div style={{position: 'absolute', left: 48, bottom: 44, display: 'flex', gap: 10, ...fadeUp(frame, 8, 12)}}>
        <Badge tone="neutral">copy-heavy edits</Badge>
        <Badge tone="neutral">literal PLD</Badge>
        <Badge tone="accent">VANTAGE</Badge>
      </div>
    </AbsoluteFill>
  );
};

const GridBackground: React.FC = () => (
  <AbsoluteFill
    style={{
      background: colors.base,
      fontFamily: font,
      overflow: 'hidden',
    }}
  >
    <svg width={960} height={540} style={{position: 'absolute', inset: 0}}>
      {Array.from({length: 21}).map((_, i) => (
        <line
          key={`v-${i}`}
          x1={i * 48}
          y1={0}
          x2={i * 48}
          y2={540}
          stroke="#111111"
          strokeWidth={1}
        />
      ))}
      {Array.from({length: 13}).map((_, i) => (
        <line
          key={`h-${i}`}
          x1={0}
          y1={i * 45}
          x2={960}
          y2={i * 45}
          stroke="#111111"
          strokeWidth={1}
        />
      ))}
    </svg>
  </AbsoluteFill>
);

export const VantageExplainer: React.FC = () => {
  return (
    <AbsoluteFill style={{background: colors.base, fontFamily: font}}>
      <GridBackground />
      <Sequence from={0} durationInFrames={72}>
        <IntroScene />
      </Sequence>
      <Sequence from={72} durationInFrames={84}>
        <BaselineScene />
      </Sequence>
      <Sequence from={156} durationInFrames={84}>
        <DriftProblemScene />
      </Sequence>
      <Sequence from={240} durationInFrames={72}>
        <FixedPromptScene />
      </Sequence>
      <Sequence from={312} durationInFrames={84}>
        <RewriteViewScene />
      </Sequence>
      <Sequence from={396} durationInFrames={72}>
        <VerificationScene />
      </Sequence>
      <Sequence from={468} durationInFrames={72}>
        <SpeedScene />
      </Sequence>
    </AbsoluteFill>
  );
};
