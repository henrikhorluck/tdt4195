extern crate gl;
extern crate nalgebra_glm as glm;
use gl::types::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{mem, os::raw::c_void, ptr, str};

mod shader;
mod util;

use glutin::event::{
    ElementState::{Pressed, Released},
    Event, KeyboardInput,
    VirtualKeyCode::{self, *},
    WindowEvent,
};
use glutin::event_loop::ControlFlow;
use std::convert::TryInto;
use std::f32::consts::PI;
use std::os::raw::c_float;

const SCREEN_W: u32 = 600;
const SCREEN_H: u32 = 600;

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
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

/// Get an offset in bytes for n units of type T
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}
// MAX_VERTEX_ATTRIBS = 16 on mac

// == // Modify and complete the function below for the first task
unsafe fn create_triangle_vao(vertices: &Vec<f32>, indices: &Vec<u32>) -> u32 {
    let mut vao_id = 0;
    let mut vbo_id = 0;
    let mut index_buffer_id = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao_id);
        gl::BindVertexArray(vao_id);

        gl::GenBuffers(1, &mut vbo_id);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_id);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            byte_size_of_array(vertices),
            pointer_to_array(vertices),
            gl::STATIC_DRAW,
        );
        // xyz of triangles
        gl::VertexAttribPointer(
            0, /*index*/
            3, /*size*/
            gl::FLOAT,
            gl::FALSE as GLboolean,
            24, /*stride*/
            ptr::null(),
        );
        // color
        gl::VertexAttribPointer(
            1,
            3,
            gl::FLOAT,
            gl::FALSE as GLboolean,
            24,
            offset::<f32>(3),
        );
        gl::EnableVertexAttribArray(0);
        gl::EnableVertexAttribArray(1);

        gl::GenBuffers(1, &mut index_buffer_id);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer_id);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            byte_size_of_array(indices),
            pointer_to_array(indices),
            gl::STATIC_DRAW,
        );
    };
    return vao_id;
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

        // == // Set up your VAO here

        /*
        PART 1 triangles
        vec![-0.4, 0.6, 0.0,  0.0, 0.4, 0.0, 0.4, 0.6, 0.0,
                                       -0.4, 0.6, 0.0, -0.4, -0.3, 0.0, -0.2, 0.15, 0.0,
                                       -0.4, -0.3, 0.0, 0.4, -0.3, 0.0, 0.0, -0.1, 0.0,
                                       0.4, -0.3, 0.0, 0.2, -0.6, 0.0, 0.4, -0.9, 0.0,
                                       -0.4, -0.9, 0.0, 0.4, -0.9, 0.0, 0.0, -0.7, 0.0,
        ];
         */
        /* CLIPPED
        vec![0.6,-0.8,-1.2,0.0,0.4,0.0,-0.8,-0.2,1.2]*/
        /*let triangles: Vec<f32> = vec![-0.4, 0.6, 0.0, 1.0, 0.0, 0.0, -0.4, 0.3, 0.0, 0.0, 1.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0,1.0,
                                       -0.4, 0.2, 0.0, 1.0, 0.0, 0.0, -0.4, -0.1, 0.0, 0.0, 1.0, 0.0, 0.4, 0.2, 0.0, 0.0, 0.0,1.0,
        ];*/
        let mut triangles: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        for i in 0..=66 {
            let step: f32 = 0.0 + (i as f32) / 66.0;
            const RADIUS: f32 = 0.5;
            // (x-x_1)^2+(y-y_1)^2-d = 0;
            let x: f32 = RADIUS * (2.0 * PI * step).cos();
            let y: f32 = RADIUS * (2.0 * PI * step).sin();
            let color: (f32, f32, f32) = match i % 3 {
                0 => (1.0, 0.0, 0.0),
                1 => (0.0, 1.0, 0.0),
                2 => (0.0, 0.0, 1.0),
                _ => panic!("Can't happen.")
            };
            let vertex: [f32; 6] = [x,y,0.0, color.0, color.1, color.2];
            triangles.append(&mut vertex.to_vec());
        }
        let triangle_count: u32 = (triangles.len() / 6) as u32;
        let mut indices: Vec<u32> = (0..=66).collect::<Vec<u32>>();
        // to complete the circle
        indices.push(1);

        let vao_id = unsafe { create_triangle_vao(&triangles, &indices) };

        // Used to demonstrate keyboard handling -- feel free to remove
        let mut _arbitrary_number = 0.0;

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
                        VirtualKeyCode::A => {
                            _arbitrary_number += delta_time;
                        }
                        VirtualKeyCode::D => {
                            _arbitrary_number -= delta_time;
                        }

                        _ => {}
                    }
                }
            }

            unsafe {
                gl::ClearColor(0.163, 0.163, 0.163, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);

                gl::UseProgram(shader.program_id);

                // Issue the necessary commands to draw your scene here
                gl::BindVertexArray(vao_id);
                gl::DrawElements(
                    gl::TRIANGLE_FAN,
                    triangle_count.try_into().unwrap(),
                    gl::UNSIGNED_INT,
                    ptr::null(),
                );

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
