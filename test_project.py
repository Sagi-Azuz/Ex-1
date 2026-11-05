"""
בדיקות ראשוניות לפרויקט הקרנת מישורים מתוך נפח תלת־ממדי.

הרצה מתוך תיקיית הפרויקט:
    py -m pytest -v test_project.py

אם קובץ המקור אינו נקרא project.py, מגדירים את הנתיב שלו ב-PowerShell:
    $env:PROJECT_FILE = ".\\Ex0.py"
    py -m pytest -v test_project.py

הערות חשובות:
1. הבדיקות טוענות את הפונקציות האמיתיות מקובץ הפרויקט, בלי להפעיל את main().
2. אם pynrrd אינו מותקן, נוצר תחליף זמני לייבוא בלבד. בקבוצת הבדיקות
   הנוכחית לא קוראים קובצי NRRD, ולכן התחליף אינו משפיע על הבדיקות.
3. Numba אינו מוחלף בתחליף. כך שגיאות התקנה או חסימה של Numba יופיעו
   בפועל ולא יוסתרו על ידי בדיקה מלאכותית.
4. הסימן EPS מתייחס לסף הדיוק הגיאומטרי שבו משתמש קוד הפרויקט.
5. אם בדיקה נכשלת, אין פירוש הדבר אוטומטית שקוד הפרויקט שגוי:
   צריך לבדוק גם שהציפייה של הבדיקה תואמת במדויק לדרישות המטלה.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import cv2

# טעינת קובץ הפרויקט בלי להפעיל את main() ובלי לדרוש ארגומנטים משורת הפקודה.
PROJECT_FILE = Path(os.environ.get("PROJECT_FILE", "project.py"))  # קביעת נתיב קובץ המקור; ברירת המחדל היא project.py.

if not PROJECT_FILE.is_absolute():
    PROJECT_FILE = Path.cwd() / PROJECT_FILE  # המרת נתיב יחסי לנתיב מלא ביחס לתיקיית העבודה הנוכחית.

if not PROJECT_FILE.exists():
    pytest.skip(
        f"Project source file was not found: {PROJECT_FILE}. "
        "Set PROJECT_FILE to the correct .py file path.",
        allow_module_level=True,
    )  # דילוג על כל קובץ הבדיקות אם קובץ המקור לא נמצא.

# בשלב הבדיקות הראשון אין צורך לקרוא קובצי NRRD.
# אם pynrrd לא מותקן, יוצרים תחליף זמני שמאפשר רק לייבא את המודול.
try:
    import nrrd  # ניסיון לייבא את ספריית pynrrd.
except ImportError:
    nrrd_stub = types.ModuleType("nrrd")  # יצירת מודול חלופי זמני לצורך ייבוא בלבד.
    nrrd_stub.read = None  # הגדרת read כערך זמני; הבדיקות הנוכחיות אינן קוראות קובצי NRRD.
    nrrd_stub.write = None  # הגדרת write כערך זמני; הבדיקות הנוכחיות אינן כותבות קובצי NRRD.
    sys.modules["nrrd"] = nrrd_stub  # רישום המודול החלופי כדי שייבוא project.py יצליח.

spec = importlib.util.spec_from_file_location("project_under_test", PROJECT_FILE)  # יצירת מפרט לטעינת קובץ הפרויקט מהנתיב שנבחר.
project = importlib.util.module_from_spec(spec)  # יצירת אובייקט מודול מתוך מפרט הטעינה.
sys.modules[spec.name] = project  # רישום המודול לפני ביצועו, כדי לאפשר למנגנון הייבוא לזהות אותו.
spec.loader.exec_module(project)  # טעינת הפונקציות והמחלקות מקובץ הפרויקט בלי להפעיל את main().


# נתוני בדיקה משותפים: נפח קטן עם ערכים צפויים מראש.
@pytest.fixture
def small_volume():  # יצירת נפח קטן שערכיו מחושבים לפי נוסחה ידועה מראש.
    z, y, x = np.indices((4, 4, 4))  # יצירת מערכי אינדקסים עבור שלושת צירי הנפח.
    return (x + 10 * y + 100 * z).astype(np.uint16)  # יצירת נפח שבו M[z,y,x] = x + 10*y + 100*z.


# בדיקות לחישוב מרחק אוקלידי בין שתי נקודות.
def test_dist_returns_zero_for_identical_points():  # בדיקה שהמרחק בין נקודה לעצמה הוא אפס.
    point = np.array([2.0, -3.0, 5.0])  # הגדרת נקודה תלת־ממדית.
    assert project.dist(point, point) == pytest.approx(0.0)  # בדיקה שהמרחק שחושב שווה לאפס.


def test_dist_calculates_3d_euclidean_distance():  # בדיקה של חישוב מרחק אוקלידי בין שתי נקודות תלת־ממדיות.
    assert project.dist(
        np.array([0.0, 0.0, 0.0]),  # נקודת ההתחלה.
        np.array([2.0, 3.0, 6.0]),  # נקודת הסיום.
    ) == pytest.approx(7.0)  # המרחק הצפוי הוא sqrt(2^2 + 3^2 + 6^2) = 7.


# בדיקות לקריאת ווקסל: הקלט הוא (x,y,z), אך סדר המערך הוא [z,y,x].
@pytest.mark.parametrize(
    "x,y,z",
    [
        (-1, 0, 0),  # בדיקת חריגה מהגבול התחתון של ציר X.
        (4, 0, 0),  # בדיקת חריגה מהגבול העליון של ציר X.
        (0, -1, 0),  # בדיקת חריגה מהגבול התחתון של ציר Y.
        (0, 4, 0),  # בדיקת חריגה מהגבול העליון של ציר Y.
        (0, 0, -1),  # בדיקת חריגה מהגבול התחתון של ציר Z.
        (0, 0, 4),  # בדיקת חריגה מהגבול העליון של ציר Z.
    ],
)
def test_get_voxel_returns_zero_outside_volume(small_volume, x, y, z):  # בדיקה שכל קואורדינטה מחוץ לגבולות הנפח מחזירה אפס.
    assert project.get_voxel(small_volume, x, y, z) == 0  # בדיקה שהפונקציה מחזירה אפס עבור כל מיקום מחוץ לנפח.


def test_get_voxel_reads_correct_xyz_location(small_volume):  # בדיקה שסדר הקואורדינטות x,y,z מותאם לסדר האחסון z,y,x.
    assert project.get_voxel(small_volume, 2, 1, 3) == 312  # הערך הצפוי הוא 2 + 10*1 + 100*3.


# בדיקות לאינטרפולציה תלת־ליניארית בין ווקסלים סמוכים.
def test_interpolation_returns_exact_value_at_integer_coordinate(small_volume):  # בדיקה שבקואורדינטות שלמות מתקבל ערך הווקסל המדויק.
    assert project.trilinear_interpolation(
        small_volume, np.array([2.0, 1.0, 3.0])  # נקודת דגימה בקואורדינטות שלמות.
    ) == pytest.approx(312.0)  # הערך הצפוי הוא ערך הווקסל באותה נקודה.


def test_interpolation_reproduces_linear_field_between_voxels(small_volume):  # בדיקה שאינטרפולציה תלת־ליניארית משחזרת שדה ליניארי בין ווקסלים.
    point = np.array([1.5, 1.5, 1.5])  # נקודת דגימה הנמצאת בין מרכזי ווקסלים.
    expected = 1.5 + 10 * 1.5 + 100 * 1.5  # חישוב הערך הצפוי לפי נוסחת השדה הליניארי.
    assert project.trilinear_interpolation(small_volume, point) == pytest.approx(expected)  # השוואת תוצאת האינטרפולציה לערך הצפוי.


def test_interpolation_at_upper_boundary_uses_zero_for_missing_neighbors(small_volume):  # בדיקה ששכנים מחוץ לנפח מקבלים ערך אפס באינטרפולציה.
    point = np.array([3.5, 1.0, 1.0])  # נקודת דגימה הנמצאת באמצע בין x=3 לבין x=4 שמחוץ לנפח.
    value_at_x3 = 3 + 10 + 100  # ערך הווקסל הקיים בגבול x=3.
    expected = 0.5 * value_at_x3  # מחצית מהמשקל שייכת לשכן החסר x=4, שערכו מוגדר כאפס.
    assert project.trilinear_interpolation(small_volume, point) == pytest.approx(expected)  # בדיקת הערך המשוקלל הצפוי.


# בדיקות לאימות מבנה הנפח, הממדים וסוג הנתונים.
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_validate_volume_accepts_supported_types(dtype):  # בדיקה שפונקציית האימות מקבלת את שלושת סוגי המספרים השלמים ללא סימן.
    volume = np.zeros((2, 3, 4), dtype=dtype)  # יצירת נפח תלת־ממדי מסוג הנתונים הנבדק.
    assert project.validate_volume(volume) is None  # נפח תקין אמור לעבור אימות ללא שגיאה.


def test_validate_volume_rejects_non_3d_array():  # בדיקה שנפח שאינו תלת־ממדי נדחה.
    with pytest.raises(ValueError, match="3D"):  # ציפייה לקבלת ValueError עם הודעה המציינת דרישה לנפח תלת־ממדי.
        project.validate_volume(np.zeros((3, 4), dtype=np.uint16))  # העברת מערך דו־ממדי לפונקציית האימות.


@pytest.mark.parametrize("shape", [(0, 2, 2), (1024, 1, 1), (1, 1024, 1), (1, 1, 1024)])
def test_validate_volume_rejects_invalid_dimensions(shape):  # בדיקה שכל ממד חייב להיות בטווח שבין 1 ל־1023.
    volume = np.zeros(shape, dtype=np.uint16)  # יצירת נפח עם ממדים שאינם עומדים בדרישות.
    with pytest.raises(ValueError, match="dimensions"):  # ציפייה לקבלת שגיאה המציינת ממדים לא תקינים.
        project.validate_volume(volume)  # בדיקת דחיית הנפח הלא תקין.


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_])
def test_validate_volume_rejects_unsupported_types(dtype):  # בדיקה שסוגי נתונים עשרוניים ובוליאניים נדחים.
    volume = np.zeros((2, 2, 2), dtype=dtype)  # יצירת נפח עם סוג הנתונים הנבדק.
    with pytest.raises(ValueError, match="data type"):  # ציפייה לקבלת שגיאה המציינת סוג נתונים לא נתמך.
        project.validate_volume(volume)  # בדיקת דחיית סוג הנתונים.


def test_validate_volume_rejects_int16():  # בדיקה שסוג הנתונים int16 נדחה בהתאם לדרישות המטלה.
    volume = np.zeros((2, 2, 2), dtype=np.int16)  # יצירת נפח מסוג int16.
    with pytest.raises(ValueError, match="data type"):  # ציפייה לקבלת שגיאה המציינת סוג נתונים לא נתמך.
        project.validate_volume(volume)  # בדיקת דחיית נפח מסוג int16.


# בדיקות תקינות גיאומטרית למישור שמוגדר באמצעות שלוש נקודות.
def test_three_point_plane_accepts_valid_non_collinear_points():  # בדיקה ששלוש נקודות שאינן קולינריות ומקיימות AB=AC מתקבלות.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (2.0, 0.0, 0.0),  # נקודת B.
        (0.0, 2.0, 0.0),  # נקודת C; המרחקים AB ו־AC שווים.
    ]
    assert bool(project.validate_plane_geometry(points)) is True  # בדיקה שהגדרת המישור תקינה.


def test_three_point_plane_rejects_collinear_points():  # בדיקה ששלוש נקודות על אותו ישר נדחות משום שאינן מגדירות מישור יחיד.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (1.0, 0.0, 0.0),  # נקודת B.
        (-1.0, 0.0, 0.0),  # נקודת C; שלוש הנקודות נמצאות על אותו ישר.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהגדרת המישור נדחית.


def test_three_point_plane_rejects_nearly_collinear_points_below_eps():  # בדיקה שמשולש שגובהו קטן מסף EPS נדחה.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (1.0, 0.0, 0.0),  # נקודת B.
        (-1.0, 0.0005, 0.0),  # נקודת C קרובה מאוד לישר AB.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהמשולש כמעט־קולינרי נדחה.


def test_three_point_plane_rejects_duplicate_points():  # בדיקה שנקודות כפולות נדחות משום שהן יוצרות צלע באורך אפס.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (1.0, 0.0, 0.0),  # נקודת B.
        (0.0, 0.0, 0.0),  # נקודת C זהה לנקודת A.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהגדרת המישור נדחית.


def test_three_point_plane_rejects_unequal_ab_ac():  # בדיקה שאורכי AB ו־AC חייבים להיות שווים בהתאם לדרישות המטלה.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (2.0, 0.0, 0.0),  # נקודת B במרחק 2 מ־A.
        (0.0, 3.0, 0.0),  # נקודת C במרחק 3 מ־A.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהמישור נדחה בגלל אורכי צלעות לא שווים.


# בדיקות תקינות גיאומטרית למישור שמוגדר באמצעות ארבע נקודות.
def test_four_point_plane_accepts_valid_coplanar_rectangle():  # בדיקה שמלבן המונח כולו במישור אחד מתקבל.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (0.0, 3.0, 0.0),  # נקודת B.
        (4.0, 0.0, 0.0),  # נקודת C.
        (4.0, 3.0, 0.0),  # נקודת D.
    ]
    assert bool(project.validate_plane_geometry(points)) is True  # בדיקה שארבע הנקודות מגדירות מישור תקין.


def test_four_point_plane_rejects_non_coplanar_points():  # בדיקה שארבע נקודות שאינן נמצאות באותו מישור נדחות.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (0.0, 3.0, 0.0),  # נקודת B.
        (4.0, 0.0, 0.0),  # נקודת C.
        (4.0, 3.0, 0.1),  # נקודת D מחוץ למישור של שלוש הנקודות האחרות.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהנקודות הלא־קופלנריות נדחות.


def test_four_point_plane_rejects_zero_length_side():  # בדיקה שנקודות כפולות היוצרות צלע באורך אפס נדחות.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (0.0, 3.0, 0.0),  # נקודת B.
        (0.0, 0.0, 0.0),  # נקודת C זהה לנקודת A.
        (4.0, 3.0, 0.0),  # נקודת D.
    ]
    assert bool(project.validate_plane_geometry(points)) is False  # בדיקה שהגדרת המישור נדחית.


# בדיקות לסידור הנקודות ולבחירת פונקציית ההקרנה המתאימה.
def test_find_valid_plane_order_returns_original_valid_order():  # בדיקה שסדר הנקודות המקורי נשמר כאשר הוא תקין.
    points = [
        (0.0, 0.0, 0.0),  # נקודת A.
        (0.0, 3.0, 0.0),  # נקודת B.
        (4.0, 0.0, 0.0),  # נקודת C.
        (4.0, 3.0, 0.0),  # נקודת D.
    ]
    assert project.find_valid_plane_order(points) == points  # בדיקה שהפונקציה מחזירה את הסדר המקורי.


def test_find_valid_plane_order_returns_none_for_collinear_three_points():  # בדיקה שסידור מחדש אינו יכול להפוך נקודות קולינריות למישור תקין.
    points = [
        (0.0, 0.0, 0.0),  # נקודה ראשונה.
        (1.0, 0.0, 0.0),  # נקודה שנייה.
        (-1.0, 0.0, 0.0),  # נקודה שלישית על אותו ישר.
    ]
    assert project.find_valid_plane_order(points) is None  # בדיקה שלא נמצא סידור תקין.


def test_project_valid_plane_rejects_unsupported_point_count(small_volume):  # בדיקה שפונקציית ההקרנה דוחה כמות נקודות שאינה שלוש או ארבע.
    with pytest.raises(ValueError, match="3 or 4"):  # ציפייה לקבלת שגיאה עבור מספר נקודות לא נתמך.
        project.project_valid_plane(small_volume, [(0, 0, 0), (1, 0, 0)])  # ניסיון להקרין מישור שהוגדר באמצעות שתי נקודות בלבד.


# בדיקות ליצירת תמונת שגיאה עבור הגדרת מישור לא תקינה.
def test_create_error_frame_returns_expected_grayscale_image():  # בדיקה שתמונת שגיאה נוצרת כתמונה דו־ממדית בגודל ובסוג הנתונים הצפויים.
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]  # נקודות לדוגמה שיוצגו בפריים השגיאה.
    frame = project.create_error_frame(7, points)  # יצירת פריים שגיאה עבור מישור מספר 7.

    assert frame.shape == (420, 700)  # בדיקת מידות תמונת השגיאה.
    assert frame.dtype == np.uint8  # בדיקה שסוג הנתונים הוא uint8.
    assert frame.max() > 0  # בדיקה שהתמונה מכילה גם פיקסלים שאינם שחורים.
    assert frame.min() == 0  # בדיקה שהתמונה מכילה רקע שחור.


# נתוני בדיקה נוספים: נפחים קבועים וליניאריים לבדיקת תוצאות ההקרנה.
@pytest.fixture
def constant_volume():  # יצירת נפח בגודל 8×8×8 שכל הווקסלים בו שווים ל־7.
    return np.full((8, 8, 8), 7, dtype=np.uint16)  # החזרת נפח קבוע מסוג uint16.


@pytest.fixture
def affine_volume():  # יצירת נפח שבו ערך כל ווקסל מחושב לפי x + 10*y + 100*z.
    z, y, x = np.indices((8, 8, 8))  # יצירת אינדקסים לכל אחד מצירי הנפח.
    return (x + 10 * y + 100 * z).astype(np.uint16)  # החזרת נפח עם שדה ליניארי ידוע מראש.


# בדיקות הקרנה באמצעות שלוש נקודות.
def test_three_point_projection_has_expected_dimensions(constant_volume):  # בדיקה שמידות ההקרנה באמצעות שלוש נקודות נכונות.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (3.0, 1.0, 1.0),  # נקודת B.
        (1.0, 3.0, 1.0),  # נקודת C.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המישור על הנפח.

    assert result.shape == (2, 3)  # בדיקת מספר השורות והעמודות של תמונת ההקרנה.
    assert result.dtype == np.float64  # בדיקה שסוג הנתונים של תוצאת ההקרנה הוא float64.


def test_three_point_projection_samples_constant_values(constant_volume):  # בדיקה שהקרנה מנפח קבוע מחזירה את הערך הקבוע או אפס.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (3.0, 1.0, 1.0),  # נקודת B.
        (1.0, 3.0, 1.0),  # נקודת C.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המישור.

    assert np.any(result == 7.0)  # בדיקה שקיימים פיקסלים שקיבלו את ערך הנפח הקבוע.
    assert np.all((result == 0.0) | (result == 7.0))  # בדיקה שכל הפיקסלים הם הערך הקבוע או אפס.


def test_three_point_projection_returns_zero_outside_volume(constant_volume):  # בדיקה שמישור הנמצא כולו מעבר לגבול X מחזיר אפסים.
    points = [
        (8.0, 1.0, 1.0),  # נקודת A מחוץ לנפח בציר X.
        (10.0, 1.0, 1.0),  # נקודת B מחוץ לנפח בציר X.
        (8.0, 3.0, 1.0),  # נקודת C מחוץ לנפח בציר X.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המישור שמחוץ לנפח.

    assert result.size > 0  # בדיקה שנוצר מערך הקרנה שאינו ריק.
    assert np.all(result == 0.0)  # בדיקה שכל ערכי ההקרנה הם אפס.


def test_three_point_projection_with_affine_volume(affine_volume):  # בדיקה שהקרנה בשדה ליניארי מחזירה ערכים מספריים תקינים.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (3.0, 1.0, 1.0),  # נקודת B.
        (1.0, 3.0, 1.0),  # נקודת C.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור על הנפח הליניארי.

    assert result.size > 0  # בדיקה שתוצאת ההקרנה אינה ריקה.
    assert np.all(np.isfinite(result))  # בדיקה שכל הערכים בתוצאה סופיים ואינם NaN או אינסוף.
    assert np.all(result >= 0.0)  # בדיקה שכל ערכי ההקרנה אינם שליליים.
    assert np.all(result <= 777.0)  # בדיקה שכל הערכים אינם חורגים מהערך המרבי בנפח הבדיקה.


# בדיקות הקרנה באמצעות ארבע נקודות.
def test_four_point_projection_has_expected_dimensions(constant_volume):  # בדיקה שמידות ההקרנה באמצעות ארבע נקודות נכונות.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (1.0, 3.0, 1.0),  # נקודת B.
        (4.0, 1.0, 1.0),  # נקודת C.
        (4.0, 3.0, 1.0),  # נקודת D.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המישור.

    assert result.shape == (3, 2)  # בדיקת מספר השורות והעמודות של תמונת ההקרנה.
    assert result.dtype == np.float64  # בדיקה שסוג הנתונים של תוצאת ההקרנה הוא float64.


def test_four_point_projection_samples_constant_values(constant_volume):  # בדיקה שהקרנה מנפח קבוע מחזירה את הערך הקבוע או אפס.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (1.0, 3.0, 1.0),  # נקודת B.
        (4.0, 1.0, 1.0),  # נקודת C.
        (4.0, 3.0, 1.0),  # נקודת D.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המישור.

    assert np.any(result == 7.0)  # בדיקה שקיימים פיקסלים שקיבלו את ערך הנפח הקבוע.
    assert np.all((result == 0.0) | (result == 7.0))  # בדיקה שכל הפיקסלים הם הערך הקבוע או אפס.


def test_four_point_projection_returns_zero_outside_volume(constant_volume):  # בדיקה שמלבן הנמצא כולו מעבר לגבול Z מחזיר אפסים.
    points = [
        (1.0, 1.0, 8.0),  # נקודת A מחוץ לנפח בציר Z.
        (1.0, 3.0, 8.0),  # נקודת B מחוץ לנפח בציר Z.
        (4.0, 1.0, 8.0),  # נקודת C מחוץ לנפח בציר Z.
        (4.0, 3.0, 8.0),  # נקודת D מחוץ לנפח בציר Z.
    ]

    result = project.project_valid_plane(constant_volume, points)  # ביצוע הקרנת המלבן שמחוץ לנפח.

    assert result.size > 0  # בדיקה שנוצר מערך הקרנה שאינו ריק.
    assert np.all(result == 0.0)  # בדיקה שכל ערכי ההקרנה הם אפס.


# בדיקות פריים שגיאה.
def test_error_frame_has_expected_shape_and_dtype():  # בדיקה שתמונת שגיאה נוצרת במידות ובסוג הנתונים הצפויים.
    frame = project.create_error_frame(
        3,  # מספר המישור שיופיע בפריים השגיאה.
        [
            (1.0, 2.0, 3.0),  # נקודה ראשונה לצורך הצגת פרטי השגיאה.
            (4.0, 5.0, 6.0),  # נקודה שנייה לצורך הצגת פרטי השגיאה.
            (7.0, 8.0, 9.0),  # נקודה שלישית לצורך הצגת פרטי השגיאה.
        ],
    )

    assert frame.shape == (420, 700)  # בדיקת מידות הפריים.
    assert frame.dtype == np.uint8  # בדיקה שסוג הנתונים הוא uint8.
    assert frame.min() == 0  # בדיקה שהרקע כולל ערך שחור.
    assert frame.max() > 0  # בדיקה שקיים בפריים תוכן שאינו שחור.


# בדיקות יצירת סרטון MP4.
def test_create_video_rejects_empty_frame_list(tmp_path):  # בדיקה שיצירת סרטון עם רשימת פריימים ריקה נכשלת.
    output = tmp_path / "empty.mp4"  # הגדרת נתיב זמני לקובץ הסרטון.

    with pytest.raises(ValueError, match="No frames"):  # ציפייה לקבלת שגיאה כאשר אין פריימים לכתיבה.
        project.create_video([], str(output), 24)  # ניסיון ליצור סרטון ללא פריימים.


@pytest.mark.parametrize("fps", [0, -1, -24])
def test_create_video_rejects_nonpositive_fps(tmp_path, fps):  # בדיקה שקצב פריימים שאינו חיובי נדחה.
    output = tmp_path / "invalid_fps.mp4"  # הגדרת נתיב זמני לקובץ הסרטון.
    frames = [np.zeros((4, 4), dtype=np.uint8)]  # יצירת פריים שחור יחיד.

    with pytest.raises(ValueError, match="FPS"):  # ציפייה לקבלת שגיאה המציינת FPS לא תקין.
        project.create_video(frames, str(output), fps)  # ניסיון ליצור סרטון עם קצב הפריימים הנבדק.


def test_create_video_writes_mp4_with_expected_dimensions_and_frames(tmp_path):  # בדיקה שסרטון MP4 נוצר עם מידות, קצב פריימים ומספר פריימים צפויים.
    output = tmp_path / "projection_test.mp4"  # הגדרת נתיב זמני לקובץ הסרטון.

    frames = [
        np.full((5, 7), 40, dtype=np.uint8),  # פריים ראשון בגודל 5×7 ובערך אפור 40.
        np.full((8, 4), 180, dtype=np.uint8),  # פריים שני בגודל 8×4 ובערך אפור 180.
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # בחירת קודק MP4V לבדיקת זמינות.
    probe = cv2.VideoWriter(
        str(output), fourcc, 12, (8, 8), isColor=True  # ניסיון לפתוח כותב וידאו בגודל 8×8 ובקצב 12 FPS.
    )

    codec_available = probe.isOpened()  # בדיקה אם הקודק זמין במערכת.
    probe.release()  # שחרור משאבי כותב הווידאו הזמני.

    if not codec_available:
        pytest.skip("MP4V codec is unavailable.")  # דילוג על הבדיקה אם קודק MP4V אינו זמין.

    if output.exists():
        output.unlink()  # מחיקת קובץ בדיקה קודם אם הוא קיים.

    project.create_video(frames, str(output), 12)  # יצירת הסרטון משני הפריימים בקצב 12 FPS.

    assert output.exists()  # בדיקה שקובץ הסרטון נוצר.
    assert output.stat().st_size > 0  # בדיקה שקובץ הסרטון אינו ריק.

    capture = cv2.VideoCapture(str(output))  # פתיחת הסרטון לקריאת מאפייניו.

    try:
        assert capture.isOpened()  # בדיקה שניתן לפתוח את הסרטון.

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # קריאת רוחב הפריימים בסרטון.
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # קריאת גובה הפריימים בסרטון.
        fps_read = capture.get(cv2.CAP_PROP_FPS)  # קריאת קצב הפריימים שנשמר בסרטון.
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # קריאת מספר הפריימים בסרטון.

        assert width % 2 == 0  # בדיקה שרוחב הסרטון זוגי, כנדרש בקודק.
        assert height % 2 == 0  # בדיקה שגובה הסרטון זוגי, כנדרש בקודק.
        assert (width, height) == (8, 8)  # בדיקה שמידות הסרטון הן 8×8 לאחר התאמת הפריימים.
        assert fps_read == pytest.approx(12, abs=1.0)  # בדיקה שקצב הפריימים קרוב ל־12 FPS.
        assert frame_count == 2  # בדיקה ששני הפריימים נכתבו לסרטון.

    finally:
        capture.release()  # שחרור משאבי קורא הווידאו גם במקרה של כישלון בבדיקה.


def test_create_video_handles_values_outside_uint8_range(tmp_path):  # בדיקה שערכים מחוץ לטווח uint8 מטופלים באמצעות חיתוך ועיגול.
    output = tmp_path / "clip_round_test.mp4"  # הגדרת נתיב זמני לקובץ הסרטון.

    frame = np.array(
        [
            [-10.0, 10.6],  # ערך שלילי וערך עשרוני לצורך בדיקת חיתוך ועיגול.
            [260.0, 100.2],  # ערך מעל 255 וערך עשרוני נוסף.
        ],
        dtype=np.float64,  # שימוש בערכי float כדי לבדוק המרה לטווח uint8.
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # בחירת קודק MP4V לבדיקת זמינות.
    probe = cv2.VideoWriter(
        str(output), fourcc, 10, (2, 2), isColor=True  # ניסיון לפתוח כותב וידאו בגודל 2×2 ובקצב 10 FPS.
    )

    codec_available = probe.isOpened()  # בדיקה אם הקודק זמין במערכת.
    probe.release()  # שחרור משאבי כותב הווידאו הזמני.

    if not codec_available:
        pytest.skip("MP4V codec is unavailable.")  # דילוג על הבדיקה אם קודק MP4V אינו זמין.

    if output.exists():
        output.unlink()  # מחיקת קובץ בדיקה קודם אם הוא קיים.

    project.create_video([frame], str(output), 10)  # יצירת סרטון מפריים יחיד עם ערכים מחוץ לטווח uint8.

    capture = cv2.VideoCapture(str(output))  # פתיחת הסרטון שנוצר לצורך בדיקת הפריים המקודד.

    try:
        assert capture.isOpened()  # בדיקה שניתן לפתוח את הסרטון.

        success, decoded = capture.read()  # קריאת הפריים הראשון מהסרטון.

        assert success  # בדיקה שהקריאה הצליחה.
        assert decoded.shape == (2, 2, 3)  # בדיקה שהפריים נקרא כתמונת BGR בגודל 2×2.

        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)  # המרת הפריים לגווני אפור לצורך בדיקת ערכי הפיקסלים.

        # MP4 הוא פורמט דחיסה מאבדת, לכן בודקים טווחים ולא שוויון מדויק.
        assert gray[0, 0] <= 8  # בדיקה שהערך השלילי נחתך בקירוב לאפס.
        assert 5 <= gray[0, 1] <= 25  # בדיקה שהערך 10.6 נשמר בקירוב לאחר עיגול ודחיסה.
        assert gray[1, 0] >= 245  # בדיקה שהערך 260 נחתך בקירוב ל־255.
        assert 85 <= gray[1, 1] <= 115  # בדיקה שהערך 100.2 נשמר בקירוב לאחר עיגול ודחיסה.

    finally:
        capture.release()  # שחרור משאבי קורא הווידאו גם במקרה של כישלון בבדיקה.


def test_three_point_projection_matches_exact_affine_values(affine_volume):  # בדיקה שערכי ההקרנה באמצעות שלוש נקודות תואמים לערכים הצפויים בשדה ליניארי.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (3.0, 1.0, 1.0),  # נקודת B.
        (1.0, 3.0, 1.0),  # נקודת C.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור על הנפח הליניארי.
    expected = np.array([
        [0.0, 113.75, 0.0],  # שורת הערכים הצפויה הראשונה.
        [115.875, 122.625, 0.0],  # שורת הערכים הצפויה השנייה.
    ], dtype=np.float64)  # שמירת הערכים הצפויים בדיוק של float64.

    np.testing.assert_allclose(
        result,
        expected,
        rtol=0.0,  # ללא סטייה יחסית מותרת.
        atol=1e-10  # מתן סטייה מוחלטת זעירה בלבד עקב חישובי נקודה צפה.
    )


def test_four_point_projection_matches_exact_affine_values(affine_volume):  # בדיקה שערכי ההקרנה באמצעות ארבע נקודות תואמים לערכים הצפויים בשדה ליניארי.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (1.0, 3.0, 1.0),  # נקודת B.
        (4.0, 1.0, 1.0),  # נקודת C.
        (4.0, 3.0, 1.0),  # נקודת D.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור על הנפח הליניארי.

    expected = np.array([
        [116.5, 126.5],  # שורת הערכים הצפויה הראשונה.
        [117.5, 127.5],  # שורת הערכים הצפויה השנייה.
        [118.5, 128.5],  # שורת הערכים הצפויה השלישית.
    ], dtype=np.float64)  # שמירת הערכים הצפויים בדיוק של float64.

    np.testing.assert_allclose(
        result,
        expected,
        rtol=0.0,  # ללא סטייה יחסית מותרת.
        atol=1e-10  # מתן סטייה מוחלטת זעירה בלבד עקב חישובי נקודה צפה.
    )


def test_three_point_projection_on_diagonal_plane(affine_volume):  # בדיקה שהקרנה של מישור אלכסוני באמצעות שלוש נקודות מחזירה מערך תקין.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (3.0, 1.0, 3.0),  # נקודת B.
        (1.0, 3.0, 3.0),  # נקודת C.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור האלכסוני.

    assert isinstance(result, np.ndarray)  # בדיקה שהתוצאה היא מערך NumPy.
    assert result.ndim == 2  # בדיקה שתמונת ההקרנה היא דו־ממדית.
    assert result.size > 0  # בדיקה שהתוצאה אינה ריקה.
    assert np.all(np.isfinite(result))  # בדיקה שכל ערכי ההקרנה סופיים.


def test_four_point_projection_on_diagonal_plane(affine_volume):  # בדיקה שהקרנה של מישור אלכסוני באמצעות ארבע נקודות מחזירה מערך תקין.
    points = [
        (1.0, 1.0, 1.0),  # נקודת A.
        (1.0, 3.0, 3.0),  # נקודת B.
        (3.0, 1.0, 3.0),  # נקודת C.
        (3.0, 3.0, 5.0),  # נקודת D.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור האלכסוני.

    assert isinstance(result, np.ndarray)  # בדיקה שהתוצאה היא מערך NumPy.
    assert result.ndim == 2  # בדיקה שתמונת ההקרנה היא דו־ממדית.
    assert result.size > 0  # בדיקה שהתוצאה אינה ריקה.
    assert np.all(np.isfinite(result))  # בדיקה שכל ערכי ההקרנה סופיים.


def test_three_point_projection_partially_outside_volume(affine_volume):  # בדיקה שהקרנה שחלקה מחוץ לנפח עדיין נוצרת ושנקודות מחוץ לנפח מקבלות אפס.
    points = [
        (-1.0, 1.0, 1.0),  # נקודת A מחוץ לנפח בציר X.
        (1.0, 1.0, 1.0),  # נקודת B בתוך הנפח.
        (-1.0, 3.0, 1.0),  # נקודת C מחוץ לנפח בציר X.
    ]

    result = project.project_valid_plane(affine_volume, points)  # ביצוע הקרנת המישור שחלקו מחוץ לנפח.

    assert result.ndim == 2  # בדיקה שתוצאת ההקרנה היא דו־ממדית.
    assert result.size > 0  # בדיקה שתוצאת ההקרנה אינה ריקה.
    assert np.any(result == 0)  # בדיקה שלפחות חלק מהפיקסלים מחוץ לנפח קיבלו ערך אפס.
# בדיקה
