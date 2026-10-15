import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const LEVEL_COLORS = { Low: 0x4c78a8, Medium: 0xb0b0b0, High: 0xe45756 };
const MESH_URL = "/fsaverage_brain.glb";

/**
 * Move each sensor marker from its scalp position (radius about 1, above the
 * cortex) straight in towards the brain's centre until it meets the cortex,
 * so the channels sit on the brain instead of floating on an invisible head.
 * HbO and HbR markers of one channel keep a small radial gap so both show.
 */
function placeMarkersOnCortex(markers, surfaces, brain) {
  const centre = new THREE.Box3().setFromObject(brain).getCenter(new THREE.Vector3());
  const raycaster = new THREE.Raycaster();
  markers.forEach((marker) => {
    const outward = marker.position.clone().sub(centre).normalize();
    raycaster.set(centre.clone().addScaledVector(outward, 3), outward.clone().negate());
    const hit = raycaster.intersectObjects(surfaces, false)[0];
    if (!hit) return;
    const lift = marker.userData.signal === "HbO" ? 0.035 : 0.09;
    marker.position.copy(hit.point).addScaledVector(outward, lift);
  });
}

export default function Brain3D({ points }) {
  const rootRef = useRef(null);
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [meshState, setMeshState] = useState("loading");

  useEffect(() => {
    const root = rootRef.current;
    const canvasHost = canvasRef.current;
    if (!root || !canvasHost) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfcf8f2);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    // Three-quarter view from the front left and above, so the shape reads as a brain.
    camera.position.set(-1.55, 1.25, -1.75);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    canvasHost.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0.15, 0.15);
    controls.minDistance = 1.2;
    controls.maxDistance = 6;

    // Strong side lighting brings out the folds of the cortex.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7a70, 1.3));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
    keyLight.position.set(-2, 3, -2);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.7);
    fillLight.position.set(2.5, 0.5, 2);
    scene.add(fillLight);

    const brainMaterial = new THREE.MeshPhongMaterial({
      color: 0xe8c4b8,
      specular: 0x332a26,
      shininess: 18,
      side: THREE.DoubleSide,
    });

    const markers = [];
    (points || []).forEach((point) => {
      if (point.x3d == null || point.y3d == null || point.z3d == null) return;
      const isHbO = point.signal === "HbO";
      const geometry = isHbO
        ? new THREE.SphereGeometry(0.038, 14, 14)
        : new THREE.BoxGeometry(0.06, 0.06, 0.06);
      const marker = new THREE.Mesh(
        geometry,
        new THREE.MeshPhongMaterial({ color: LEVEL_COLORS[point.level] || 0x999999 })
      );
      const scale = isHbO ? 0.975 : 1.025;
      marker.position.set(point.x3d * scale, point.z3d * scale, -point.y3d * scale);
      marker.userData = point;
      scene.add(marker);
      markers.push(marker);
    });

    const loader = new GLTFLoader();
    let disposed = false;
    loader.load(
      MESH_URL,
      (gltf) => {
        if (disposed) return;
          const surfaces = [];
        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.material = brainMaterial;
            child.geometry.computeVertexNormals();
            surfaces.push(child);
          }
        });
        scene.add(gltf.scene);
        placeMarkersOnCortex(markers, surfaces, gltf.scene);
        setMeshState("ready");
      },
      undefined,
      () => {
        if (!disposed) setMeshState("error");
      }
    );

    const resize = () => {
      const width = Math.max(1, canvasHost.clientWidth);
      const height = Math.max(240, Math.min(width, 520));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvasHost);
    resize();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPointerMove = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(markers)[0];
      if (!hit) {
        setTooltip(null);
        return;
      }
      setTooltip({
        left: event.clientX - rect.left + 12,
        top: event.clientY - rect.top + 12,
        point: hit.object.userData,
      });
    };
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", () => setTooltip(null));
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material && object.material !== brainMaterial) {
          object.material.dispose();
        }
      });
      brainMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [points]);

  return (
    <div className="brain3d" ref={rootRef}>
      <div className="brain3d-canvas" ref={canvasRef} />
      {tooltip && (
        <div className="brain3d-tooltip" style={{ left: tooltip.left, top: tooltip.top }}>
          <strong>{tooltip.point.bare_channel}</strong> ({tooltip.point.signal})
          <br />Level: {tooltip.point.level}
          <br />Region: {tooltip.point.functional_region}
          <br />Rule {tooltip.point.rule_id} -&gt; {tooltip.point.consequent}
          <br />ACC={tooltip.point.accuracy != null ? Number(tooltip.point.accuracy).toFixed(2) : "n/a"}
        </div>
      )}
      {meshState === "loading" && <p className="caption">Loading the brain model…</p>}
      {meshState === "error" && (
        <p className="brain3d-status">
          The brain model could not be loaded, so only the sensor positions are shown.
        </p>
      )}
      <p className="caption">
        Each channel is drawn where it meets the brain surface (a standard brain,
        fsaverage). Drag to rotate, scroll to zoom.
      </p>
    </div>
  );
}
