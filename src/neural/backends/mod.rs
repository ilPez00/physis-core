//! Feature-gated backends (P3-P9). The deterministic backend lives in
//! `crate::neural::substrate` and needs no feature; these do.

pub mod reservoir;
pub mod spiking;
pub mod recorded;
pub mod fly;
pub mod cortical;

pub use reservoir::RandomReservoirBackend;
pub use spiking::ArtificialSpikingBackend;
pub use recorded::RecordedNeuralBackend;
pub use fly::FlyConnectomeBackend;
pub use cortical::CorticalSimulatorBackend;
