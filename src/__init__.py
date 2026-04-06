"""
src/__init__.py  — SurgicalAI v3 package exports
"""

from .extractor      import (extract_coordinates_from_folder,
                              extract_instrument_position,
                              process_entire_archive)
from .metrics        import (calculate_all_metrics, stability_timeline,
                              compute_baseline, percentile_rank,
                              phase_time_analysis)
from .classifier     import (classify, train_classifier,
                              skill_color, skill_badge_color, skill_emoji,
                              skill_clinical_equiv, score_band, score_colour)
from .feedback       import generate_feedback
from .report_generator import generate_pdf_report
from .visualizer     import (draw_trajectory_on_frame, create_expert_comparison,
                              find_best_frame)
from .phase_reader   import (load_phase_annotations, get_dominant_phase,
                              find_phase_file)
from .video_processor import (download_video_frames, extract_frames_from_video)

__all__ = [
    "extract_coordinates_from_folder", "extract_instrument_position",
    "process_entire_archive",
    "calculate_all_metrics", "stability_timeline",
    "compute_baseline", "percentile_rank", "phase_time_analysis",
    "classify", "train_classifier",
    "skill_color", "skill_badge_color", "skill_emoji", "skill_clinical_equiv",
    "score_band", "score_colour",
    "generate_feedback",
    "generate_pdf_report",
    "draw_trajectory_on_frame", "create_expert_comparison", "find_best_frame",
    "load_phase_annotations", "get_dominant_phase", "find_phase_file",
    "download_video_frames", "extract_frames_from_video",
]
