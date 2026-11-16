import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const status = document.querySelector('#status');
try {
  const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  document.body.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202323);
  const camera = new THREE.PerspectiveCamera(40, 1, .001, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x707080, 2.5));
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(-1, 2, 3); scene.add(light);
  const group = new THREE.Group(); scene.add(group);
  let meshes = [], revision = -1, initialized = false;
  function resize() {
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  }
  function reset() {
    const box = new THREE.Box3();
    for (const mesh of meshes) if(mesh.visible) box.expandByObject(mesh);
    if(box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length()/2, .05);
    const fov = Math.min(camera.fov*Math.PI/180, 2*Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(0,0,radius/Math.sin(fov/2)*1.35));
    controls.update();
  }
  function visibility() {
    for(const mesh of meshes) {
      mesh.visible = document.getElementById(mesh.userData.side).checked;
      mesh.material.wireframe = document.querySelector('#wire').checked;
    }
  }
  for(const id of ['right','left','wire']) document.getElementById(id).onchange=visibility;
  document.querySelector('#reset').onclick=reset;
  window.addEventListener('resize',resize); resize();
  async function poll() {
    try {
      const response = await fetch('./state', {cache:'no-store'});
      if(!response.ok) throw new Error('disconnected');
      const data = await response.json();
      status.textContent = data.hands.length ? '' : '当前帧没有有效手部参数';
      if(data.revision !== revision) {
        revision = data.revision;
        for(const mesh of meshes) {group.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}
        meshes = data.hands.map(hand => {
          const geometry = new THREE.BufferGeometry();
          // Rotate the master-camera axes into a conventional Y-up view.
          const points = hand.vertices.flatMap(([x,y,z]) => [x,-y,-z]);
          geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
          geometry.setIndex(hand.faces.flat());geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({
            color:hand.side==='right'?0x67d7b1:0xe5a2ce,roughness:.65,side:THREE.DoubleSide}));
          mesh.userData.side=hand.side;group.add(mesh);return mesh;
        });
        document.querySelector('#frame').textContent = `帧 ${data.frame}`;
        visibility();
        if(!initialized && meshes.length) {reset();initialized=true;}
      }
    } catch(error) {status.textContent='编辑器已断开连接';}
    setTimeout(poll,200);
  }
  poll();
  renderer.setAnimationLoop(() => {controls.update();renderer.render(scene,camera);});
} catch(error) {status.textContent = '三维视图启动失败：' + error.message;}
