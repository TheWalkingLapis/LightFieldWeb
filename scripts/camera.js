class Camera {
  ngp_factor = 4.03112885717555/1.5;
  radius = 1.5;

  theta_min = 0.01;
  theta_max = 0.49 * Math.PI;
  phi_min = -Math.PI;
  phi_max = Math.PI;

  theta_sensitivity = 0.005;
  phi_sensitivity = 0.01;

  constructor(theta, phi) {
    phi += -0.5 * Math.PI; // rotate such that chair is ff, scene dependant though
    this.theta = Math.min(this.theta_max, Math.max(this.theta_min, theta));
    this.phi = Math.min(this.phi_max, Math.max(this.phi_min, phi));
    this.c2w = this.c2w_sphere(this.theta, this.phi, this.radius);
  }

  rotate(delta_theta, delta_phi) {
    // clamp theta
    this.theta += delta_theta;
    this.theta = Math.min(this.theta_max, Math.max(this.theta_min, this.theta));
    // wrap phi
    this.phi += delta_phi;
    if (this.phi > this.phi_max) {
      this.phi -= (this.phi_max - this.phi_min);
    } else if (this.phi < this.phi_min) {
      this.phi += (this.phi_max - this.phi_min);
    }
    this.phi = Math.min(this.phi_max, Math.max(this.phi_min, this.phi));
    this.c2w = this.c2w_sphere(this.theta, this.phi, this.radius);
  }

  normalize(vec) {
    const len = Math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2);
    return vec.map(v => v / len);
  }
  cross(a, b) {
    return [
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    ];
  }
  c2w_to_input(c2w) {
    const c2w_in = [
        [c2w[0][0], -c2w[0][1], -c2w[0][2], this.ngp_factor * c2w[0][3]],
        [c2w[1][0], -c2w[1][1], -c2w[1][2], this.ngp_factor * c2w[1][3]],
        [c2w[2][0], -c2w[2][1], -c2w[2][2], this.ngp_factor * c2w[2][3]]
    ];
    return c2w_in
  }
  c2w_sphere(theta, phi, radius) {
    const x = radius * Math.sin(theta) * Math.cos(phi);
    const y = radius * Math.sin(theta) * Math.sin(phi);
    const z = radius * Math.cos(theta);

    const upW = [0, 0, -1];
    const f = this.normalize([-x, -y, -z]);
    const r = this.normalize(this.cross(upW, f));
    const u = this.cross(f, r);

    const R = [
      [r[0], u[0], f[0]],
      [r[1], u[1], f[1]],
      [r[2], u[2], f[2]],
    ];

    const c2w = [
      [R[0][0], R[0][1], R[0][2], x],
      [R[1][0], R[1][1], R[1][2], y],
      [R[2][0], R[2][1], R[2][2], z],
    ];

    return c2w;
  }

  get_c2w_as_input() {
    return this.c2w_to_input(this.c2w);
  }

  get_position() {
    return this.c2w.map(row => [row[3]]).flat();
  }

  mousedown_hook(event) {
    this.isDragging = true;
    this.pendingUpdate = false;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
  }
  mouseup_hook(event) {
    this.isDragging = false;
  }
  mousemove_hook(event, render_hook) {
    if (!this.isDragging) return;

    const dx = event.clientX - this.lastX;
    const dy = event.clientY - this.lastY;
    this.lastX = event.clientX;
    this.lastY = event.clientY;

    this.rotate(-dy * this.theta_sensitivity, -dx * this.phi_sensitivity);
    if (!this.pendingUpdate) {
      this.pendingUpdate = true;
      requestAnimationFrame(async () => {
        await render_hook();
        this.pendingUpdate = false;
      });
    }
  }
}