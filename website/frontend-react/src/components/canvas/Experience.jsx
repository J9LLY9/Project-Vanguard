import { PerspectiveCamera } from "@react-three/drei";
import DustSystem from "./DustSystem";
import SingularitySphere from "./SingularitySphere";
import TaglineParticles from "./TaglineParticles";
import SupernovaFlash from "./SupernovaFlash";
import VanguardNodeLogo from "./VanguardNodeLogo";
import QualityController from "./QualityController";

/**
 * All of Acts 1-3's 3D content, in one Canvas — see components/PinnedActs.jsx,
 * which owns the actual <Canvas> and the GSAP scroll-jack pin. JSX ORDER
 * matters here: <SingularitySphere> is declared AFTER <DustSystem> so its
 * render-to-texture backdrop capture runs once the dust simulation has
 * already advanced for the current frame — see SingularitySphere.jsx's
 * Teacher Mode note on why that ordering falls out of declaration order
 * with no custom useFrame priority needed.
 */
export default function Experience({ particleConfig }) {
  return (
    <>
      <QualityController />
      <PerspectiveCamera makeDefault position={[0, 0, 9]} fov={50} />

      <DustSystem count={particleConfig.dust} textureSize={particleConfig.textureSize} />
      <SingularitySphere />
      <TaglineParticles />
      <SupernovaFlash />
      <VanguardNodeLogo />
    </>
  );
}
