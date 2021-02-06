extern crate gl;
extern crate nalgebra_glm as glm;

use gl::types::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr};

mod mesh;
mod scene_graph;
mod shader;
mod toolbox;
mod util;

use crate::scene_graph::SceneNode;
use glutin::event::{
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;
use std::convert::TryInto;
use std::f32::consts::PI;
use std::pin::Pin;

const SCREEN_W: u32 = 600;
const SCREEN_H: u32 = 600;

struct Camera {
    /// location in x-direction
    x: f32,
    y: f32,
    z: f32,
    /// yaw (left-right
    yaw: f32,
    pitch: f32,
    roll: f32,
}

// Helper functions to make interacting with OpenGL a little bit prettier. You will need these!
// The names should be pretty self explanatory
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

/// Get the OpenGL-compatible pointer to an arbitrary array of numbers
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

/// Get the size of the given type in bytes
#[allow(dead_code)]
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

/// Get an offset in bytes for n units of type T
#[allow(dead_code)]
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}
// MAX_VERTEX_ATTRIBS = 16 on mac

// == // Modify and complete the function below for the first task
unsafe fn create_triangle_vao(
    vertices: &Vec<f32>,
    rgba: &Vec<f32>,
    indices: &Vec<u32>,
    normals: &Vec<f32>,
) -> u32 {
    let mut vao_id = 0;
    let mut vbo_ids: [u32; 3] = [0, 0, 0];
    let mut index_buffer_id: u32 = 0;

    const VERTEX_INDEX: u32 = 0;
    const COLOR_INDEX: u32 = 1;
    const NORMAL_INDEX: u32 = 2;

    gl::GenVertexArrays(1, &mut vao_id);
    gl::BindVertexArray(vao_id);

    // create vertex buffer objecta
    gl::GenBuffers(3, vbo_ids.as_mut_ptr());

    // bind so that it can be modified
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_ids[VERTEX_INDEX as usize]);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(vertices),
        pointer_to_array(vertices),
        gl::STATIC_DRAW,
    );

    // xyz of triangles
    gl::VertexAttribPointer(
        VERTEX_INDEX, /*index*/
        3,            /*size*/
        gl::FLOAT,
        gl::FALSE as GLboolean,
        0, /*stride*/
        ptr::null(),
    );

    gl::EnableVertexAttribArray(VERTEX_INDEX);

    // create vertex buffer object
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_ids[COLOR_INDEX as usize]);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(rgba),
        pointer_to_array(rgba),
        gl::STATIC_DRAW,
    );

    // color
    gl::VertexAttribPointer(
        COLOR_INDEX,
        4,
        gl::FLOAT,
        gl::FALSE as GLboolean,
        0,
        ptr::null(),
    );
    gl::EnableVertexAttribArray(COLOR_INDEX);

    gl::BindBuffer(gl::ARRAY_BUFFER, vbo_ids[NORMAL_INDEX as usize]);
    gl::BufferData(
        gl::ARRAY_BUFFER,
        byte_size_of_array(normals),
        pointer_to_array(normals),
        gl::STATIC_DRAW,
    );

    // color
    gl::VertexAttribPointer(
        NORMAL_INDEX,
        3,
        gl::FLOAT,
        gl::FALSE as GLboolean,
        0,
        ptr::null(),
    );
    gl::EnableVertexAttribArray(NORMAL_INDEX);

    gl::GenBuffers(1, &mut index_buffer_id);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer_id);
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER,
        byte_size_of_array(indices),
        pointer_to_array(indices),
        gl::STATIC_DRAW,
    );

    return vao_id;
}

fn rotate_around_point(rotation: &glm::Vec3, reference_point: &glm::Vec3) -> glm::Mat4 {
    let translated_from_ref = glm::translate(&glm::identity(), &reference_point);
    let rotate_x: glm::Mat4 = glm::rotation(rotation.x, &glm::vec3(1.0, 0.0, 0.0));
    let rotate_y: glm::Mat4 = glm::rotation(rotation.y, &glm::vec3(0.0, 1.0, 0.0));
    let rotate_z: glm::Mat4 = glm::rotation(rotation.z, &glm::vec3(0.0, 0.0, 1.0));
    let translate_to_ref = glm::translate(&glm::identity(), &-reference_point);

    return translated_from_ref * rotate_z * rotate_y * rotate_x * translate_to_ref;
}

unsafe fn update_node_transformations(
    root: &mut scene_graph::SceneNode,
    transformation_so_far: &glm::Mat4,
) {
    // Construct the correct transformation matrix
    // Update the node's transformation matrix
    root.current_transformation_matrix = transformation_so_far
        * glm::translation(&root.position)
        * rotate_around_point(&root.rotation, &root.reference_point);

    // Recurse
    for &child in &root.children {
        update_node_transformations(&mut *child, &root.current_transformation_matrix)
    }
}

unsafe fn draw_scene(root: &scene_graph::SceneNode, view_projection_matrix: &glm::Mat4) {
    // Check if node is drawable, set uniforms, draw

    let t_mat = view_projection_matrix * &root.current_transformation_matrix;

    if root.index_count != -1 {
        gl::UniformMatrix4fv(3, 1, gl::FALSE, t_mat.as_ptr());
        gl::UniformMatrix4fv(4, 1, gl::FALSE, root.current_transformation_matrix.as_ptr());
        gl::BindVertexArray(root.vao_id);
        gl::DrawElements(
            gl::TRIANGLES,
            root.index_count,
            gl::UNSIGNED_INT,
            ptr::null(),
        );
    }

    // Recurse
    for &child in &root.children {
        draw_scene(&*child, &view_projection_matrix);
    }
}

fn spin_node(node: &mut scene_graph::SceneNode, speed: f32, elapsed_time: f32, axis: usize) {
    node.rotation[axis] += speed * elapsed_time;
}

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(false)
        .with_inner_size(glutin::dpi::LogicalSize::new(SCREEN_W, SCREEN_H));
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Send a copy of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers. This has to be done inside of the renderin thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        // Set up openGL
        unsafe {
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            // Activate Z-buffering
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            //
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());
        }

        // Basic usage of shader helper
        // The code below returns a shader object, which contains the field .program_id
        // The snippet is not enough to do the assignment, and will need to be modified (outside of just using the correct path)
        let shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.frag")
                .attach_file("./shaders/simple.vert")
                .link()
        };

        // set up scene
        let mesh = mesh::Terrain::load("./resources/lunarsurface.obj");
        let vao_id = unsafe { mesh.to_vao() };
        let mut root_node = SceneNode::new();
        let mut terrain_node = SceneNode::from_vao(vao_id, mesh.index_count);
        root_node.add_child(&terrain_node);

        let heli = mesh::Helicopter::load("./resources/helicopter.obj");
        let heli_body_id = unsafe { heli.body.to_vao() };
        let heli_door_id = unsafe { heli.door.to_vao() };
        let heli_main_rotor_id = unsafe { heli.main_rotor.to_vao() };
        let heli_tail_rotor_id = unsafe { heli.tail_rotor.to_vao() };

        const NUMBER_OF_HELICOPTERS: usize = 20;
        // very ugly type-signature, but is needed for `.collect()` to work
        let mut helis: Vec<mem::ManuallyDrop<Pin<Box<SceneNode>>>> = (0..NUMBER_OF_HELICOPTERS).map(|_| SceneNode::from_vao(heli_body_id, heli.body.index_count)).collect();
        let mut heli_tail_rotors: Vec<mem::ManuallyDrop<Pin<Box<SceneNode>>>> = (0..NUMBER_OF_HELICOPTERS).map(|_| {
            let mut node = SceneNode::from_vao(heli_tail_rotor_id, heli.tail_rotor.index_count);
            node.reference_point = glm::vec3(0.35, 2.3, 10.4);
            node
        }).collect();
        let mut heli_main_rotors: Vec<mem::ManuallyDrop<Pin<Box<SceneNode>>>> = (0..NUMBER_OF_HELICOPTERS).map(|_| SceneNode::from_vao(heli_main_rotor_id, heli.main_rotor.index_count)).collect();
        let mut heli_doors: Vec<mem::ManuallyDrop<Pin<Box<SceneNode>>>> = (0..NUMBER_OF_HELICOPTERS).map(|_| SceneNode::from_vao(heli_door_id, heli.door.index_count)).collect();

        // add helicopters
        helis.iter().for_each(|node| terrain_node.add_child(&node));

        for (idx,   body_node) in (&mut helis).iter_mut().enumerate() {
            // heli_tail_rotor_node.reference_point = glm::vec3(0.35, 2.3, 10.4);
            // heli_main_rotor_node.position = glm::vec3(0.0,-3.0,0.0);
            // heli_main_rotor_node.reference_point = glm::vec3(3.0, -2.0, 5.0);
            body_node.add_child(&heli_doors[idx]);
            body_node.add_child(&heli_main_rotors[idx]);
            body_node.add_child(&heli_tail_rotors[idx]);
            // heli_body_node.rotation = glm::vec3(0.0, 0.5, 0.0);
            // heli_body_node.reference_point = glm::vec3(1.0,4.0,1.0);
        }

        let mut camera = Camera {
            x: 1.0,
            y: 1.0,
            // Start with -1 since view-box is from -1 to 1, this way we see the scene at the beginning
            z: -1.0,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        };

        let first_frame_time = std::time::Instant::now();
        let mut last_frame_time = first_frame_time;
        // The main rendering loop
        loop {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(last_frame_time).as_secs_f32();
            last_frame_time = now;

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        VirtualKeyCode::W => {
                            camera.z += delta_time * camera.yaw.cos();
                            camera.x -= delta_time * camera.yaw.sin();
                        }
                        VirtualKeyCode::S => {
                            camera.z -= delta_time * camera.yaw.cos();
                            camera.x += delta_time * camera.yaw.sin();
                        }
                        VirtualKeyCode::A => {
                            camera.x += delta_time * camera.yaw.cos();
                            camera.z += delta_time * camera.yaw.sin();
                        }
                        VirtualKeyCode::D => {
                            camera.x -= delta_time * camera.yaw.cos();
                            camera.z -= delta_time * camera.yaw.sin();
                        }
                        VirtualKeyCode::E => {
                            camera.y -= delta_time;
                        }
                        VirtualKeyCode::Q => {
                            camera.y += delta_time;
                        }
                        VirtualKeyCode::Left => {
                            camera.yaw -= delta_time;
                        }
                        VirtualKeyCode::Right => {
                            camera.yaw += delta_time;
                        }
                        VirtualKeyCode::Up => {
                            camera.pitch -= delta_time;
                        }
                        VirtualKeyCode::Down => {
                            camera.pitch += delta_time;
                        }
                        VirtualKeyCode::N => {
                            camera.roll += delta_time;
                        }
                        VirtualKeyCode::M => {
                            camera.roll -= delta_time;
                        }
                        _ => {}
                    }
                }
            }

            unsafe {
                // gl::ClearColor(0.163, 0.163, 0.163, 1.0);
                gl::ClearColor(1.0, 1.0, 1.0, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                gl::UseProgram(shader.program_id);

                // let rotation: glm::Mat4 = glm::rotation(elapsed, &glm::vec3(0.0,1.0,0.0));

                let perspective: glm::Mat4 =
                    glm::perspective((SCREEN_H as f32) / (SCREEN_W as f32), 45.0, 1.0, 10000.0);

                // translate scene relative to camera
                let mat_xyz: glm::Mat4 =
                    glm::translation(&glm::vec3(1.0 * camera.x, 1.0 * camera.y, 1.0 * camera.z));
                // roll = rotation around z-axis (NOTE: this makes it kinda trippy, since movement is not compensated)
                let mat_roll: glm::Mat4 = glm::rotation(camera.roll, &glm::vec3(0.0, 0.0, 1.0));
                // pitch = rotation around x-axis
                let mat_pitch: glm::Mat4 = glm::rotation(camera.pitch, &glm::vec3(1.0, 0.0, 0.0));
                // yaw = rotation around y-axis
                let mat_yaw: glm::Mat4 = glm::rotation(camera.yaw, &glm::vec3(0.0, 1.0, 0.0));

                // camera.yaw, pitch, roll = 2pi gives full rotation

                // first translate the position, then the rotations (
                let mat4 = perspective * mat_pitch * mat_yaw * mat_roll * mat_xyz;

                // send matrix to shader
                //gl::UniformMatrix4fv(3, 1, gl::FALSE, mat4.as_ptr());

                for main_rotor_node in heli_main_rotors.iter_mut() {
                    main_rotor_node.rotation.y = 20.0 * elapsed;
                }
                for heli_tail_rotor in heli_tail_rotors.iter_mut() {
                    heli_tail_rotor.rotation.x = 10.0 * elapsed;
                }
                for (idx, heli) in helis.iter_mut().enumerate() {
                    let heading = toolbox::simple_heading_animation( elapsed);
                    heli.position.x = heading.x + idx as f32 * 15.0;
                    heli.position.z = heading.z;
                    heli.rotation.x = heading.pitch;
                    heli.rotation.y = heading.yaw;
                    heli.rotation.z = heading.roll;
                }

                // heli_body_node.rotation.y = PI / 2.0; // task 5 a)

                // Issue the necessary commands to draw your scene here

                unsafe { update_node_transformations(&mut root_node, &glm::identity()) }
                unsafe { draw_scene(&root_node, &mat4) };

                gl::UseProgram(0);
            }

            context.swap_buffers().unwrap();
        }
    });

    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events get handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: key_state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        }
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle escape separately
                match keycode {
                    Escape => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    });
}
