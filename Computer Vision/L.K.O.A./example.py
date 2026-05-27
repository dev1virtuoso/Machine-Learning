import time
from module import (
    ProductionSystemConfig,
    ThreadSafeCameraReader,
    RobustUARTCommandPublisher,
    ProductionVisionEngine,
    VehicleState
)

if __name__ == "__main__":
    sys_config = ProductionSystemConfig(
        production_mode=True,
        target_fps=12,
        uart_port="/dev/ttyAMA0"
    )

    hardware_reader = ThreadSafeCameraReader(sys_config).start()
    control_publisher = RobustUARTCommandPublisher(sys_config)

    time.sleep(2.0)

    engine = ProductionVisionEngine(sys_config)
    
    frame_interval_sec = 1.0 / sys_config.target_fps
    print("[RUNNING] Production-Grade Edge Vision System successfully launched on hardware.")

    try:
        while True:
            loop_start_time = time.time()
            
            frame_fetched, active_frame = hardware_reader.get_latest_frame()
            if not frame_fetched or active_frame is None:
                time.sleep(0.002)
                continue
                
            current_state, linear_v, steer_w = engine.process_frame(active_frame)
            
            control_publisher.transmit(current_state, linear_v, steer_w)
            
            if not sys_config.production_mode:
                print(f"[DEBUG LOG] State: {current_state.name} | V: {linear_v:.2f} | W: {steer_w:.2f}")
            
            execution_cost = time.time() - loop_start_time
            remaining_sleep = frame_interval_sec - execution_cost
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)
                
    except KeyboardInterrupt:
        print("\n[TERMINATION] Shutdown intercept issued. Safely cycling down system modules...")
    finally:
        hardware_reader.terminate()
        control_publisher.close()
        print("[TERMINATION] Hardware pipelines severed cleanly. Safe exit finalized.")