import math  # ספרייה לפעולות מתמטיות, כגון שורש ריבועי, עיגול וחישוב רצפה.
import numpy as np  # ספרייה לעבודה עם מערכים וחישובים מספריים.
import nrrd  # ספרייה לקריאה של נפחים תלת־ממדיים מקובצי NRRD.
import csv  # ספרייה לקריאת קובץ ה-CSV המכיל את הגדרות המישורים.
from itertools import permutations  # יצירת כל הסידורים האפשריים של נקודות המישור.
import time  # מדידת זמני ביצוע של פעולות בתוכנית.
import cv2  # ספרייה לעיבוד תמונות וליצירת סרטון MP4.
from numba import njit, prange  # האצת פונקציות חישוביות באמצעות הידור JIT ואפשרות להרצה מקבילית.


@njit(cache=True)  # האצת הפונקציה באמצעות Numba ושמירת הגרסה המהודרת במטמון.
def dist(p1,p2):
    """
    מטרת הפונקציה:
    לחשב את המרחק האוקלידי בין שתי נקודות במרחב תלת־ממדי.

    קלט:
    p1 - הנקודה הראשונה, עם קואורדינטות X, Y, Z.
    p2 - הנקודה השנייה, עם קואורדינטות X, Y, Z.

    פלט:
    המרחק המספרי בין שתי הנקודות.

    אופן הפעולה:
    מחושב ההפרש בין הקואורדינטות בכל ציר, כל הפרש מועלה בריבוע,
    שלוש התוצאות מחוברות ומחושב מהסכום שורש ריבועי.
    """
    dx=p2[0]-p1[0]  # חישוב ההפרש בין הקואורדינטות בציר X.
    dy=p2[1]-p1[1]  # חישוב ההפרש בין הקואורדינטות בציר Y.
    dz=p2[2]-p1[2]  # חישוב ההפרש בין הקואורדינטות בציר Z.
    return math.sqrt(dx*dx+dy*dy+dz*dz)  # החזרת המרחק האוקלידי בין הנקודות.


@njit(cache=True)  # האצת הגישה לווקסלים באמצעות Numba.
def get_voxel(M,x,y,z):
    """
    מטרת הפונקציה:
    להחזיר את ערך הווקסל במיקום המבוקש בתוך הנפח.

    קלט:
    M - מערך תלת־ממדי המכיל את הנפח.
    x, y, z - אינדקסי הווקסל המבוקש.

    פלט:
    ערך הווקסל אם המיקום נמצא בתוך הנפח; אחרת מוחזר אפס.

    אופן הפעולה:
    הפונקציה בודקת את גבולות הנפח לפני הגישה למערך,
    וכך מונעת גישה לאינדקסים שאינם קיימים.
    """
    sz,sy,sx=M.shape  # קבלת ממדי הנפח בסדר Z, Y, X.
    if x>=sx or x<0 or y>=sy or y<0 or z>=sz or z<0:  # בדיקה אם אחד האינדקסים נמצא מחוץ לגבולות הנפח.
        return 0  # החזרת אפס אם המיקום מחוץ לנפח.
    else:  # המיקום נמצא בתוך גבולות הנפח.
        return M[z,y,x]  # החזרת ערך הווקסל במיקום המבוקש.


@njit(cache=True)  # האצת חישובי האינטרפולציה באמצעות Numba.
def trilinear_interpolation(M,P):
    """
    מטרת הפונקציה:
    לחשב ערך משוער בנקודה תלת־ממדית באמצעות אינטרפולציה תלת־ליניארית.

    קלט:
    M - מערך הנפח התלת־ממדי.
    P - נקודת הדגימה בעלת הקואורדינטות X, Y, Z.

    פלט:
    ערך מספרי המשוער בנקודת הדגימה.

    אופן הפעולה:
    הפונקציה מאתרת את שמונת הווקסלים המקיפים את נקודת הדגימה,
    מחשבת לכל אחד מהם משקל בהתאם למרחק היחסי מהנקודה,
    ומחברת את ערכי הווקסלים לאחר הכפלה במשקליהם.
    אם הנקודה נמצאת בדיוק באינדקסים שלמים, מוחזר ערך הווקסל
    ישירות ללא חישוב אינטרפולציה.
    """
    x,y,z=P  # חילוץ הקואורדינטות של נקודת הדגימה.
    x0=math.floor(x)  # מציאת האינדקס השלם התחתון בציר X.
    x1=x0+1  # מציאת האינדקס הבא בציר X.
    y0=math.floor(y)  # מציאת האינדקס השלם התחתון בציר Y.
    y1=y0+1  # מציאת האינדקס הבא בציר Y.
    z0=math.floor(z)  # מציאת האינדקס השלם התחתון בציר Z.
    z1=z0+1  # מציאת האינדקס הבא בציר Z.
    dx=x-x0  # חישוב החלק היחסי בין האינדקסים בציר X.
    dy=y-y0  # חישוב החלק היחסי בין האינדקסים בציר Y.
    dz=z-z0  # חישוב החלק היחסי בין האינדקסים בציר Z.
    if dx==0.0 and dy==0.0 and dz==0.0:  # בדיקה אם כל הקואורדינטות שלמות.
        return get_voxel(M,x0,y0,z0)  # החזרת ערך הווקסל ישירות ללא אינטרפולציה.
    # חישוב המשקלים של שמונת הווקסלים המקיפים את נקודת הדגימה.
    w000=(1-dx)*(1-dy)*(1-dz)  # חישוב משקל הפינה הראשונה.
    w001=(1-dx)*(1-dy)*dz  # חישוב משקל הפינה השנייה.
    w010=(1-dx)*dy*(1-dz)  # חישוב משקל הפינה השלישית.
    w011=(1-dx)*dy*dz  # חישוב משקל הפינה הרביעית.
    w100=dx*(1-dy)*(1-dz)  # חישוב משקל הפינה החמישית.
    w101=dx*(1-dy)*dz  # חישוב משקל הפינה השישית.
    w110=dx*dy*(1-dz)  # חישוב משקל הפינה השביעית.
    w111=dx*dy*dz  # חישוב משקל הפינה השמינית.
    # קריאת ערכי שמונת הווקסלים מהנפח.
    c000=get_voxel(M,x0,y0,z0)  # קריאת ערך הווקסל הראשון.
    c001=get_voxel(M,x0,y0,z1)  # קריאת ערך הווקסל השני.
    c010=get_voxel(M,x0,y1,z0)  # קריאת ערך הווקסל השלישי.
    c011=get_voxel(M,x0,y1,z1)  # קריאת ערך הווקסל הרביעי.
    c100=get_voxel(M,x1,y0,z0)  # קריאת ערך הווקסל החמישי.
    c101=get_voxel(M,x1,y0,z1)  # קריאת ערך הווקסל השישי.
    c110=get_voxel(M,x1,y1,z0)  # קריאת ערך הווקסל השביעי.
    c111=get_voxel(M,x1,y1,z1)  # קריאת ערך הווקסל השמיני.
    # חישוב הערך המשוער באמצעות סכום ערכי הווקסלים המשוקללים.
    return (w000*c000+w001*c001+w010*c010+w011*c011+w100*c100+w101*c101+w110*c110+w111*c111)


@njit(cache=True,parallel=True)  # האצת ליבת ההקרנה ושימוש בהרצה מקבילית של שורות התמונה.
def project_plane_core(M,a,b,c,d,is_four_points):
    """
    מטרת הפונקציה:
    לחשב תמונת הקרנה דו־ממדית של מישור המוגדר בתוך נפח תלת־ממדי.

    קלט:
    M - מערך הנפח התלת־ממדי.
    a, b, c, d - נקודות המגדירות את המישור.
    is_four_points - דגל המציין אם המישור מוגדר באמצעות ארבע נקודות.

    פלט:
    מטריצה דו־ממדית E המכילה את ערכי ההקרנה.

    אופן הפעולה:
    תחילה נקבעים ממדי תמונת ההקרנה לפי המרחקים בין הנקודות.
    לאחר מכן הפונקציה עוברת על שורות התמונה, מחשבת את קצות
    כל שורה ומדגמת נקודות לאורכה. ערך כל נקודה מחושב באמצעות
    אינטרפולציה תלת־ליניארית. נקודות מחוץ לנפח מקבלות ערך אפס.
    לבסוף, ערכי הדגימה נשמרים במטריצת ההקרנה.
    """
    sz,sy,sx=M.shape  # קבלת ממדי הנפח בצירים Z, Y, X.
    # קביעת ממדי תמונת ההקרנה בהתאם למספר הנקודות.
    if is_four_points:  # בדיקה אם המישור מוגדר באמצעות ארבע נקודות.
        H=max(1,round(dist(a,c)))  # קביעת גובה התמונה לפי המרחק בין A ל-C.
        W=max(1,round(dist(c,d)))  # קביעת רוחב התמונה לפי המרחק בין C ל-D.
    else:  # המישור מוגדר באמצעות שלוש נקודות.
        H=max(1,round(dist(a,b)))  # קביעת גובה התמונה לפי המרחק בין A ל-B.
        W=max(1,round(dist(b,c)))  # קביעת רוחב התמונה לפי המרחק בין B ל-C.
    E=np.zeros((H,W),dtype=np.float64)  # יצירת מטריצת הקרנה בגודל H על W, המאותחלת באפס.
    for j in prange(H):  # מעבר מקבילי על שורות תמונת ההקרנה.
        u=(j+0.5)/H  # חישוב מיקום יחסי לאורך הגובה, במרכז כל מקטע דגימה.
        # חישוב שתי נקודות הקצה של השורה הנוכחית.
        if is_four_points:  # בחירת הגאומטריה המתאימה למישור בעל ארבע נקודות.
            A1=a+u*(c-a)  # חישוב הקצה הראשון באמצעות אינטרפולציה בין A ל-C.
            A2=b+u*(d-b)  # חישוב הקצה השני באמצעות אינטרפולציה בין B ל-D.
        else:  # בחירת הגאומטריה המתאימה למישור בעל שלוש נקודות.
            A1=a+u*(b-a)  # חישוב הקצה הראשון בין A ל-B.
            A2=a+u*(c-a)  # חישוב הקצה השני בין A ל-C.
        d12=dist(A1,A2)  # חישוב המרחק בין קצות השורה.
        N=max(1,round(d12))  # קביעת מספר הדגימות לאורך השורה.
        start_x=math.floor((W-N)/2)  # חישוב נקודת ההתחלה האופקית למרכוז הדגימות.

        for i in range(N):  # מעבר על הדגימות לאורך השורה.
            v=(i +0.5)/N  # חישוב המיקום היחסי של הדגימה לאורך השורה.
            P=A1+v*(A2-A1)  # חישוב נקודת הדגימה במרחב התלת־ממדי.
            px,py,pz=P  # חילוץ קואורדינטות נקודת הדגימה.
            # בדיקה אם נקודת הדגימה נמצאת בתוך גבולות הנפח.
            if (px<0 or px>=sx or py<0 or py>=sy or pz<0 or pz>=sz):
                value=0.0  # קביעת ערך אפס לנקודה שנמצאת מחוץ לנפח.
            else:  # נקודת הדגימה נמצאת בתוך גבולות הנפח.
                value=trilinear_interpolation(M,P)  # חישוב ערך הדגימה באמצעות אינטרפולציה תלת־ליניארית.
            x=start_x+i  # חישוב המיקום האופקי של הדגימה בתמונת ההקרנה.
            if 0<=x<W:  # בדיקה שהמיקום נמצא בתוך רוחב התמונה.
                E[j,x]=value  # שמירת ערך הדגימה בתא המתאים במטריצת ההקרנה.
    return E  # החזרת תמונת ההקרנה שחושבה.


@njit(cache=True)  # האצת פונקציית ההקרנה עבור מישורים בעלי שלוש נקודות.
def project_plane_3(M,a,b,c):
    """
    מטרת הפונקציה:
    לבצע הקרנה של מישור המוגדר באמצעות שלוש נקודות.

    קלט:
    M - הנפח התלת־ממדי.
    a, b, c - שלוש הנקודות המגדירות את המישור.

    פלט:
    מטריצת ההקרנה הדו־ממדית.

    אופן הפעולה:
    הנקודות מומרות למערכי float64, נוצרת נקודת עזר d,
    ולאחר מכן מופעלת פונקציית הליבה במצב של שלוש נקודות.
    """
    a=np.asarray(a,dtype=np.float64)  # המרת הנקודה הראשונה למערך מספרי מסוג float64.
    b=np.asarray(b,dtype=np.float64)  # המרת הנקודה השנייה למערך מספרי מסוג float64.
    c=np.asarray(c,dtype=np.float64)  # המרת הנקודה השלישית למערך מספרי מסוג float64.
    d=np.zeros(3,dtype=np.float64)  # יצירת נקודת עזר כדי להתאים לממשק של פונקציית הליבה.
    return project_plane_core(M,a,b,c,d,False)  # הפעלת ליבת ההקרנה במצב של שלוש נקודות והחזרת התוצאה.


@njit(cache=True)  # האצת פונקציית ההקרנה עבור מישורים בעלי ארבע נקודות.
def project_plane_4(M,a,b,c,d):
    """
    מטרת הפונקציה:
    לבצע הקרנה של מישור המוגדר באמצעות ארבע נקודות.

    קלט:
    M - הנפח התלת־ממדי.
    a, b, c, d - ארבע הנקודות המגדירות את המישור.

    פלט:
    מטריצת ההקרנה הדו־ממדית.

    אופן הפעולה:
    כל ארבע הנקודות מומרות למערכי float64, ולאחר מכן מופעלת
    פונקציית הליבה במצב של ארבע נקודות.
    """
    a=np.asarray(a,dtype=np.float64)  # המרת הנקודה הראשונה למערך מספרי מסוג float64.
    b=np.asarray(b,dtype=np.float64)  # המרת הנקודה השנייה למערך מספרי מסוג float64.
    c=np.asarray(c,dtype=np.float64)  # המרת הנקודה השלישית למערך מספרי מסוג float64.
    d=np.asarray(d,dtype=np.float64)  # המרת הנקודה הרביעית למערך מספרי מסוג float64.
    return project_plane_core(M,a,b,c,d,True)  # הפעלת ליבת ההקרנה במצב של ארבע נקודות והחזרת התוצאה.


def validate_volume(M):
    """
    מטרת הפונקציה:
    לבדוק שהנפח שנטען עומד בדרישות התוכנית.

    קלט:
    M - מערך הנפח התלת־ממדי.

    פלט:
    אין ערך מוחזר. אם הנפח אינו תקין, הפונקציה מעלה ValueError.

    אופן הפעולה:
    נבדקים מספר הממדים, גודל כל ממד וסוג הנתונים של המערך.
    """
    if M.ndim!=3:  # בדיקה שהנפח מכיל בדיוק שלושה ממדים.
        raise ValueError("Volume must be a 3D matrix")  # עצירת התוכנית אם הנפח אינו תלת־ממדי.
    sz,sy,sx=M.shape  # קבלת ממדי הנפח בסדר Z, Y, X.
    if not all(1<=size<2**10 for size in (sx,sy,sz)):  # בדיקה שכל ממד נמצא בין 1 ל-1023.
        raise ValueError(  # העלאת שגיאה אם אחד ממדי הנפח אינו תקין.
            f"Invalid volume dimensions: x={sx}, y={sy}, z={sz}. " 
            "Each dimension must be between 1 and 1023.")
    # הגדרת סוגי הנתונים שהפונקציה יודעת לקבל.
    allowed_types=(np.dtype("uint8"),np.dtype("uint16"),np.dtype("uint32"))
    if M.dtype not in allowed_types:  # בדיקה שסוג הנתונים של הנפח נמצא ברשימה המותרת.
        raise ValueError(  # העלאת שגיאה אם סוג הנתונים אינו נתמך.
            f"Invalid volume data type: {M.dtype}. " 
            "Expected uint8, uint16 or uint32.")


def validate_plane_geometry(points):
    """
    מטרת הפונקציה:
    לבדוק אם הגדרת מישור עומדת בתנאים הגאומטריים המוגדרים בקוד.

    קלט:
    points - רשימה המכילה שלוש או ארבע נקודות תלת־ממדיות.

    פלט:
    True אם התנאים מתקיימים, או False אם ההגדרה אינה תקינה.

    אופן הפעולה:
    עבור שלוש נקודות נבדקים שוויון המרחקים AB ו-AC ואורכים חיוביים.
    עבור ארבע נקודות נבדקים שוויון הצלעות, סדר הבסיסים,
    אורכים חיוביים וקו־מישוריות של הנקודות.
    """
    EPS=0.001  # הגדרת סף ההשוואה המספרי לבדיקות הגאומטריות.
    if len(points) == 3:  # בדיקת מישור המוגדר על ידי שלוש נקודות
        # המרת הנקודות למערכי NumPy לצורך חישובים וקטוריים
        a,b,c=[np.asarray(p, dtype=float) for p in points]
        # חישוב המרחקים בין כל זוג נקודות
        dab=dist(a,b)  # המרחק בין A ל-B
        dac=dist(a,c)  # המרחק בין A ל-C
        dbc=dist(b,c)  # המרחק בין B ל-C
        print(f"  dab = {dab:.6f}, dac = {dac:.6f}, dbc = {dbc:.6f}")
        print(f"  |dab - dac| = {abs(dab - dac):.6f}")
        # בדיקה שהמרחקים AB ו-AC שווים בקירוב לפי EPS
        valid_equal_sides=abs(dab-dac)<EPS
        # בדיקה שכל המרחקים בין הנקודות חיוביים
        valid_lengths=dab>0 and dac>0 and dbc>0
        # חישוב המכפלה הווקטורית של AB ו-AC
        # אם שלוש הנקודות נמצאות על אותו קו, גודל המכפלה יהיה אפס
        cross_product=np.cross(b-a,c-a)
        # חישוב שטח המקבילית שנוצרת על ידי שני הווקטורים
        parallelogram_area=np.linalg.norm(cross_product)
        # חישוב אורך הצלע הארוכה ביותר לצורך קבלת גובה גאומטרי
        longest_side=max(dab,dac,dbc)
        # חלוקת שטח המקבילית באורך הצלע הארוכה נותנת גובה
        # כך ניתן להשוות את הגובה לסף EPS, שהוא סף של מרחק
        valid_non_collinear =(longest_side>0 and parallelogram_area/longest_side>EPS)
        # הדפסת תוצאות הבדיקות לצורך איתור שגיאות
        print(f"  Equal sides: {valid_equal_sides}")
        print(f"  Positive lengths: {valid_lengths}")
        print(f"  Non-collinear points: {valid_non_collinear}")
        # המישור תקין רק אם כל התנאים מתקיימים
        return (valid_equal_sides and valid_lengths and valid_non_collinear)
    if len(points)==4:  # בדיקה אם המישור מוגדר באמצעות ארבע נקודות.
        a1,a2,b,c=[np.asarray(p,dtype=float) for p in points]  # המרת ארבע הנקודות למערכים מספריים.
        side1=dist(a1,b)  # חישוב אורך הצלע הראשונה.
        side2=dist(a2,c)  # חישוב אורך הצלע השנייה.
        base1=dist(a1,a2)  # חישוב אורך הבסיס הראשון.
        base2=dist(b,c)  # חישוב אורך הבסיס השני.
        # חישוב מדד הקו־מישוריות באמצעות המכפלה המשולשת הסקלרית.
        coplanarity=abs(np.dot(np.cross(a2-a1,b-a1),c-a1))
        print(f"  side1 = {side1:.6f}, side2 = {side2:.6f}")  # הצגת אורכי הצלעות.
        print(f"  base1 = {base1:.6f}, base2 = {base2:.6f}")  # הצגת אורכי הבסיסים.
        print(f"  |side1 - side2| = {abs(side1 - side2):.6f}")  # הצגת ההפרש בין הצלעות.
        print(f"  Coplanarity value = {coplanarity:.6f}")  # הצגת מדד הקו־מישוריות.
        valid_equal_sides=abs(side1-side2)<EPS  # בדיקה אם אורכי הצלעות שווים בקירוב.
        valid_bases=base1<=base2  # בדיקה שסדר אורכי הבסיסים עומד בתנאי הקוד.
        valid_lengths=(side1>0 and side2>0 and base1>0 and base2>0)  # בדיקה שכל האורכים חיוביים.
        valid_coplanarity=coplanarity<EPS  # בדיקה שמדד הקו־מישוריות קטן מהסף.
        print(f"  Equal sides: {valid_equal_sides}")  # הצגת תוצאת בדיקת שוויון הצלעות.
        print(f"  Base order: {valid_bases}")  # הצגת תוצאת בדיקת סדר הבסיסים.
        print(f"  Positive lengths: {valid_lengths}")  # הצגת תוצאת בדיקת האורכים.
        print(f"  Coplanar: {valid_coplanarity}")  # הצגת תוצאת בדיקת הקו־מישוריות.
        return (  # החזרת התוצאה המשולבת של כל הבדיקות.
            valid_equal_sides  # התנאי הראשון: שוויון צלעות.
            and valid_bases  # התנאי השני: סדר בסיסים תקין.
            and valid_lengths  # התנאי השלישי: אורכים חיוביים.
            and valid_coplanarity  # התנאי הרביעי: קו־מישוריות.
)
    print(f"  Invalid number of points: {len(points)}")  # הצגת מספר הנקודות הלא תקין.
    return False  # דחיית ההגדרה כאשר מספר הנקודות אינו שלוש או ארבע.


def find_valid_plane_order(points):
    """
    מטרת הפונקציה:
    למצוא סדר נקודות שעומד בתנאים הגאומטריים של התוכנית.

    קלט:
    points - רשימת נקודות המישור.

    פלט:
    רשימת נקודות בסדר תקין, או None אם לא נמצא סדר מתאים.

    אופן הפעולה:
    תחילה נבדק הסדר המקורי. אם הוא אינו תקין, הפונקציה
    עוברת על תמורות הנקודות ובודקת כל סידור עד למציאת סידור תקין.
    """
    # בדיקת הסדר המקורי לפני חיפוש סידורים אחרים.
    if validate_plane_geometry(points):  # בדיקה אם הסדר המקורי תקין.
        return points  # החזרת הרשימה המקורית אם היא תקינה.
    # חיפוש סידור חלופי רק אם הסדר המקורי אינו תקין.
    for candidate in permutations(points):  # יצירת התמורות האפשריות של הנקודות.
        candidate=list(candidate)  # המרת התמורה לרשימה.
        if candidate==points:  # בדיקה אם התמורה זהה לרשימה המקורית שכבר נבדקה.
            continue  # דילוג לתמורה הבאה.
        if validate_plane_geometry(candidate):  # בדיקה אם הסידור החדש תקין.
            return candidate  # החזרת הסידור הראשון שעבר את הבדיקה.
    return None  # החזרת None אם לא נמצא אף סידור תקין.


def project_valid_plane(M, points):
    """
    מטרת הפונקציה:
    לבחור את פונקציית ההקרנה המתאימה למספר הנקודות.

    קלט:
    M - הנפח התלת־ממדי.
    points - שלוש או ארבע נקודות המגדירות מישור.

    פלט:
    מטריצת ההקרנה הדו־ממדית.

    אופן הפעולה:
    אם יש שלוש נקודות, מופעלת project_plane_3.
    אם יש ארבע נקודות, מופעלת project_plane_4.
    כל מספר אחר גורם להעלאת שגיאה.
    """
    if len(points)==3:  # בדיקה אם המישור מוגדר באמצעות שלוש נקודות.
        a,b,c=points  # חילוץ שלוש הנקודות מהרשימה.
        return project_plane_3(M,a,b,c)  # ביצוע הקרנה באמצעות פונקציית שלוש הנקודות.
    if len(points)==4:  # בדיקה אם המישור מוגדר באמצעות ארבע נקודות.
        a,b,c,d=points  # חילוץ ארבע הנקודות מהרשימה.
        return project_plane_4(M,a,b,c,d)  # ביצוע הקרנה באמצעות פונקציית ארבע הנקודות.

    raise ValueError("A plane must contain 3 or 4 points.")  # העלאת שגיאה אם מספר הנקודות אינו נתמך.


def create_error_frame(plane_number, points):
    """
    מטרת הפונקציה:
    ליצור פריים שגיאה עבור מישור שאינו תקין.

    קלט:
    plane_number - מספר המישור לפי סדרו בקובץ CSV.
    points - הנקודות המקוריות שהתקבלו מקובץ CSV.

    פלט:
    תמונת שגיאה בגווני אפור הכוללת את מספר המישור ואת הקואורדינטות שלו.

    אופן הפעולה:
    נוצרת תמונה שחורה בגודל קבוע.
    על התמונה נכתבת כותרת המציינת את מספר המישור ואת העובדה שהוא אינו תקין.
    לאחר מכן מוצגות הקואורדינטות המקוריות של כל נקודות המישור.
    """
    frame=np.zeros((420,700),dtype=np.uint8)  # יצירת תמונה שחורה בגודל 700 על 420 פיקסלים.
    # הצגת כותרת השגיאה על גבי התמונה.
    cv2.putText(frame, f"PLANE {plane_number}-ERROR",(25, 55),cv2.FONT_HERSHEY_SIMPLEX,1.0,255,2)
    # הצגת הודעה המציינת שהמישור אינו תקין.
    cv2.putText(frame,"INVALID PLANE",(25,105),cv2.FONT_HERSHEY_SIMPLEX,0.8,255,2)
    # הצגת כל הנקודות המקוריות שהתקבלו עבור המישור.
    for i,point in enumerate(points):
        # בניית מחרוזת הכוללת את מספר הנקודה ואת הקואורדינטות שלה.
        point_text=(f"P{i + 1} = "f"({point[0]:.3f},{point[1]:.3f},{point[2]:.3f})")
        # כתיבת הקואורדינטות על גבי פריים השגיאה.
        cv2.putText(frame,point_text,(25,165+i*50),cv2.FONT_HERSHEY_SIMPLEX,0.65,255,2)
    return frame  # החזרת תמונת השגיאה.


def create_video(projections, output_file, fps):
    """
    מטרת הפונקציה:
    ליצור סרטון MP4 שבו כל הקרנה או פריים שגיאה מוצגים
    כפריים אחד שלם.

    קלט:
    projections - רשימת פריימים לסרטון. הרשימה יכולה להכיל
                  הקרנות תקינות וגם פריימי שגיאה.
    output_file - נתיב קובץ הווידאו שייוצר.
    fps - מספר הפריימים לשנייה בסרטון.

    פלט:
    קובץ MP4 שבו כל פריים מופיע לפי הסדר שבו התקבל ברשימה.

    אופן הפעולה:
    תחילה נקבע גודל אחיד לכל פריימי הסרטון לפי הפריים הגדול ביותר.
    לאחר מכן כל פריים מומר לתמונה בגווני אפור, ממורכז בתוך
    פריים בגודל האחיד ומומר לפורמט BGR.
    כל פריים נכתב לקובץ הווידאו.
    הפריימים נשמרים מראש ברשימה בזיכרון על ידי הפונקציה main.
    """
    if not projections:  # בדיקה אם רשימת הפריימים ריקה.
        raise ValueError("No frames available for the video.")  # העלאת שגיאה אם אין פריימים לסרטון.
    if fps<=0:  # בדיקה שקצב הפריימים חיובי.
        raise ValueError("FPS must be greater than zero.")  # העלאת שגיאה אם קצב הפריימים אינו תקין.
    # קביעת הגודל האחיד של הסרטון לפי הפריים הגדול ביותר ברשימה.
    max_height=max(p.shape[0] for p in projections)  # מציאת גובה הפריים הגדול ביותר.
    max_width=max(p.shape[1] for p in projections)  # מציאת רוחב הפריים הגדול ביותר.
    # התאמת הממדים למספרים זוגיים, כנדרש על ידי קודקי MP4 נפוצים.
    video_height=max_height+(max_height%2)  # התאמת גובה הסרטון למספר זוגי.
    video_width=max_width+(max_width%2)  # התאמת רוחב הסרטון למספר זוגי.
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")  # בחירת קודק לקידוד הסרטון בפורמט MP4V.
    # יצירת כותב הווידאו עם הנתיב, הקודק, קצב הפריימים וגודל הסרטון.
    writer=cv2.VideoWriter(output_file,fourcc,fps,(video_width,video_height),isColor=True)
    if not writer.isOpened():  # בדיקה אם כותב הווידאו נפתח בהצלחה.
        raise RuntimeError(f"Could not create video: {output_file}")  # העלאת שגיאה אם לא ניתן ליצור את הקובץ.
    total_frames=0  # אתחול מונה הפריימים שנכתבו לסרטון.
    try:  # התחלת בלוק שמבטיח שחרור משאבים גם במקרה של שגיאה.
        # מעבר על כל הפריימים לפי הסדר שבו הם מופיעים ברשימה.
        for plane_number,projection in enumerate(projections,start=1):
            height,width=projection.shape  # קבלת גובה ורוחב הפריים הנוכחי.
            # יצירת פריים שחור בגודל האחיד של הסרטון.
            frame=np.zeros((video_height,video_width),dtype=np.uint8)
            # עיגול ערכי הפריים, הגבלתם לטווח 0–255 והמרתם ל-8 ביט.
            image=np.clip(np.rint(projection), 0, 255).astype(np.uint8)
            y_offset=(video_height-height)//2  # חישוב ההיסט האנכי למרכוז התמונה.
            x_offset=(video_width-width)//2  # חישוב ההיסט האופקי למרכוז התמונה.
            # הצבת התמונה המלאה במרכז הפריים השחור.
            frame[y_offset:y_offset+height,x_offset:x_offset+width]=image
            # המרת הפריים מגווני אפור לפורמט BGR בעל שלושה ערוצים.
            color_frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            writer.write(color_frame)  # כתיבת הפריים לקובץ הווידאו.
            total_frames+=1  # הגדלת מונה הפריימים לאחר כתיבת פריים אחד.
            # הצגת התקדמות כתיבת הפריימים לסרטון.
            print(f"Plane {plane_number}/{len(projections)} " f"written; total frames: {total_frames}")
    finally:  # בלוק שמתבצע גם במקרה של שגיאה בזמן כתיבת הסרטון.
        writer.release()  # שחרור כותב הווידאו וסגירת הקובץ.
    # הצגת פרטי הסרטון שנוצר.
    print(f"Video saved to: {output_file}")  # הצגת נתיב קובץ הווידאו.
    print(f"Video dimensions: {video_width} x {video_height}")  # הצגת ממדי הסרטון.
    print(f"Video FPS: {fps}")  # הצגת קצב הפריימים שנקבע.
    print(f"Video frames: {total_frames}")  # הצגת מספר הפריימים שנכתבו.
    print(f"Video duration: {total_frames/fps:.2f} seconds")  # חישוב והצגת משך הסרטון בשניות.


def main():
    """
    מטרת הפונקציה:
    לנהל את כל שלבי הפעלת התוכנית, החל מקריאת הקבצים
    ועד יצירת הסרטון והצגת נתוני הביצועים.

    קלט:
    שלושה ארגומנטים משורת הפקודה:
    קובץ NRRD, קובץ CSV ונתיב לסרטון MP4.

    פלט:
    יצירת סרטון ההקרנות והדפסת מידע על הקלט, ההקרנות והביצועים.

    אופן הפעולה:
    1. קריאת הנפח והמטא־נתונים.
    2. בדיקת הנפח והמרת סוג הנתונים.
    3. קריאת המישורים מקובץ CSV.
    4. בדיקת תקינות גאומטרית וסידור הנקודות.
    5. יצירת פריימי שגיאה עבור מישורים לא תקינים.
    6. חישוב ההקרנות התקינות ומדידת זמני הביצוע.
    7. יצירת סרטון MP4 לפי סדר המישורים המקורי.
    8. הצגת סיכום הביצועים ובדיקת יעדי הקצב.
    """
    import argparse  # ייבוא הספרייה לקריאת ארגומנטים משורת הפקודה.
    # יצירת מנתח הארגומנטים של התוכנית.
    parser=argparse.ArgumentParser(description="Project planes onto a 3D volume and create an MP4 video.")
    parser.add_argument("nrrd_file",help="Input NRRD volume")  # הגדרת הארגומנט של קובץ הנפח.
    parser.add_argument("csv_file",help="CSV file containing plane definitions")  # הגדרת הארגומנט של קובץ המישורים.
    parser.add_argument("output_file",help="Output MP4 video path")  # הגדרת הארגומנט של קובץ הפלט.
    args=parser.parse_args()  # קריאת הארגומנטים שהועברו בעת הפעלת התוכנית.
    # טעינת הנפח התלת־ממדי והמטא־נתונים מקובץ NRRD.
    M,header=nrrd.read(args.nrrd_file,index_order="C")
    # הצגת מידע על הנפח ועל מערכת הקואורדינטות שלו.
    print("Volume shape:",M.shape)  # הצגת ממדי הנפח.
    print("NRRD space:",header.get("space"))  # הצגת מערכת הקואורדינטות של הנפח.
    print("Space origin:",header.get("space origin"))  # הצגת ראשית מערכת הקואורדינטות.
    print("Space directions:\n",header.get("space directions"))  # הצגת וקטורי הכיוון של הצירים.
    print("Volume dtype:",M.dtype)  # הצגת סוג הנתונים המקורי של הנפח.
    print("Volume value range:",M.min(),M.max())  # הצגת ערכי הנפח המינימלי והמקסימלי.
    # הודעה המציינת שקואורדינטות ה-CSV צריכות להתאים למערכת הקואורדינטות של הנפח.
    print("CSV plane coordinates must match the volume coordinate system.")
    if M.size==0:  # בדיקה אם המערך אינו מכיל איברים.
        raise ValueError("The input volume is empty.")  # עצירת התוכנית אם הנפח ריק.
    # בדיקה שערכי הנפח נמצאים בטווח שאפשר לייצג באמצעות uint16.
    if M.min()<0 or M.max()>np.iinfo(np.uint16).max:
        raise ValueError("Volume values cannot be represented as uint16")  # עצירת התוכנית אם הערכים אינם בטווח המותר.
    M=M.astype(np.uint16)  # המרת ערכי הנפח לסוג uint16 לצורך המשך העיבוד.
    validate_volume(M)  # בדיקת תקינות ממדי הנפח וסוג הנתונים לאחר ההמרה.
    planes=[]  # יצירת רשימה לאחסון הגדרות המישורים מקובץ ה-CSV.
    # פתיחת קובץ ה-CSV לקריאה, תוך תמיכה בקידוד UTF-8 עם BOM.
    with open(args.csv_file, newline="",encoding="utf-8-sig") as csvfile:
        reader=csv.reader(csvfile)  # יצירת אובייקט לקריאת שורות הקובץ.
        next(reader,None)  # דילוג על שורת הכותרת של קובץ ה-CSV.
        # מעבר על שורות הקובץ, כאשר מספר השורה מתחיל מ-2 בגלל הכותרת.
        for row_number,row in enumerate(reader,start=2):
            # בדיקה אם השורה ריקה או מכילה רווחים בלבד.
            if not row or all(not value.strip() for value in row):
                continue  # דילוג לשורה הבאה.
            try:  # התחלת טיפול בשגיאות שעלולות להתרחש בעת קריאת השורה.
                num_points=int(row[0])  # קריאת מספר הנקודות המוגדרות בשורה.
                if num_points not in (3,4):  # בדיקה שמספר הנקודות הוא שלוש או ארבע.
                    raise ValueError("A plane must contain 3 or 4 points.")  # העלאת שגיאה אם מספר הנקודות אינו נתמך.
                # קריאת הקואורדינטות והמרתן למספרים עשרוניים.
                values=[float(value) for value in row[1:1+num_points*3]]
                # בדיקה שמספר הקואורדינטות תואם למספר הנקודות שהוגדר.
                if len(values)!=num_points*3:
                    raise ValueError("Incorrect number of coordinate values.")  # העלאת שגיאה אם מספר הקואורדינטות שגוי.
                # חלוקת הקואורדינטות לקבוצות של שלושה ערכים: X, Y, Z.
                points=[tuple(values[i:i+3]) for i in range(0,len(values),3)]
                planes.append(points)  # הוספת הגדרת המישור לרשימת המישורים.
            except (ValueError,IndexError) as error:  # תפיסת שגיאות המרה או גישה לאיברים בשורה.
                raise ValueError(f"Invalid CSV row {row_number}: {error}") from error  # הצגת מספר השורה שבה נמצאה השגיאה.

    if not planes:  # בדיקה אם לא נמצאו הגדרות מישורים בקובץ.
        raise ValueError("The CSV file contains no plane definitions.")  # עצירת התוכנית אם אין מישורים.
    # יצירת רשימת משימות שתשמור על הסדר המקורי של המישורים בקובץ CSV.
    plane_jobs=[]
    # מעבר על כל המישורים ובדיקת התקינות הגאומטרית שלהם.
    for i,points in enumerate(planes,start=1):
        # בדיקת הסדר המקורי וחיפוש סדר חלופי אם הסדר המקורי אינו תקין.
        ordered_points=find_valid_plane_order(points)
        if ordered_points is None:  # בדיקה אם לא נמצא סידור תקין של נקודות המישור.
            print(f"Plane {i}: invalid - Error Frame created")
            # יצירת פריים שגיאה המציג את מספר המישור ואת הקואורדינטות המקוריות.
            error_frame = create_error_frame(i,points)
            # שמירת פריים השגיאה במיקום המתאים ברצף הסרטון.
            plane_jobs.append(("error",error_frame))
        else:  # נמצא סדר נקודות תקין עבור המישור.
            if ordered_points!=points:  # בדיקה אם היה צורך לשנות את סדר הנקודות.
                print(f"Plane {i}: valid - point order updated")
            else:  # הסדר המקורי נמצא תקין.
                print(f"Plane {i}: valid")
            # שמירת הנקודות התקינות לצורך חישוב ההקרנה בהמשך.
            plane_jobs.append(("valid",ordered_points))
    # שלב 1: חישוב ההקרנות התקינות ומדידת הביצועים שלהן.
    # חימום Numba לפני מדידת זמן חישוב ההקרנות.
    # מטרת החימום היא לבצע קומפילציה של פונקציות ההקרנה לפני המדידה.
    # כך זמן הקומפילציה הראשוני לא ייכלל בזמן הביצוע הנמדד.
    # יצירת נקודות מלאכותיות עבור מישור המוגדר באמצעות 3 נקודות.
    dummy_3_points=[(-12.75,8.35,-3.5),(4.25,-16.8,7.65),(-9.4,-5.25,13.2)]
    # יצירת נקודות מלאכותיות עבור מישור המוגדר באמצעות 4 נקודות.
    dummy_4_points=[(-12.75,8.35,-3.5),(4.25,-16.8,7.65),(-9.4,-5.25,13.2),(15.6,-11.45,-8.25)]
    # הפעלת שתי פונקציות ההקרנה לצורך קומפילציה מראש.
    _ = project_plane_3(M,*dummy_3_points)  # חימום פונקציית ההקרנה עבור שלוש נקודות.
    _ = project_plane_4(M,*dummy_4_points)  # חימום פונקציית ההקרנה עבור ארבע נקודות.
    print("Numba warm-up completed for both plane types.")  # הצגת הודעה על סיום שלב החימום.
    # התחלת מדידת זמן ההקרנות לאחר פעולת החימום.
    projection_start=time.perf_counter()  # שמירת זמן ההתחלה למדידת חישוב ההקרנות.
    projections=[]  # רשימת הקרנות תקינות בלבד, המשמשת לחישוב נתוני הביצועים.
    video_frames=[]  # רשימת כל הפריימים, כולל הקרנות ופריימי שגיאה, לפי הסדר המקורי.
    total_pixels=0  # אתחול מונה הפיקסלים של ההקרנות התקינות בלבד.
    # מעבר על כל משימות המישורים לפי הסדר המקורי.
    for plane_number,(job_type,data) in enumerate(plane_jobs,start=1):
        if job_type=="error":  # בדיקה אם המשימה מכילה פריים שגיאה.
            # הוספת פריים השגיאה למיקומו המקורי בסרטון.
            video_frames.append(data)
            continue  # מעבר למישור הבא ללא חישוב הקרנה וללא הכללתו במדידת FPS.
        points=data  # שמירת נקודות המישור התקין במשתנה נוח לשימוש.
        plane_start=time.perf_counter()  # שמירת זמן ההתחלה של חישוב ההקרנה הנוכחית.
        # חישוב ההקרנה הדו־ממדית של המישור התקין.
        projection=project_valid_plane(M,points)
        plane_elapsed=time.perf_counter()-plane_start  # חישוב זמן ההקרנה של המישור הנוכחי.
        # שמירת ההקרנה ברשימת ההקרנות התקינות וברשימת פריימי הסרטון.
        projections.append(projection)
        video_frames.append(projection)
        height, width=projection.shape  # קבלת גובה ורוחב תמונת ההקרנה.
        total_pixels+=height*width  # הוספת מספר הפיקסלים של ההקרנה למונה הכולל.
        # חישוב קצב ההקרנות של המישור הנוכחי.
        plane_fps=1.0/plane_elapsed if plane_elapsed>0 else 0.0
        # הצגת ממדי ההקרנה וזמן החישוב שלה.
        print(f"Plane {plane_number}: {height} rows x {width} columns")
        print(f"  Projection time: {plane_elapsed:.6f} seconds")
        print(f"  Projection FPS: {plane_fps:.2f}")
    # חישוב זמן הטיפול הכולל במשימות ההקרנה.
    # זמן זה כולל את מעבר הלולאה ואת הטיפול בפריימי שגיאה.
    projection_elapsed=time.perf_counter()-projection_start
    number_of_projections=len(projections)  # ספירת מספר ההקרנות התקינות שחושבו.
    # חישוב מספר ההקרנות התקינות לשנייה.
    # פריימי שגיאה אינם נכללים במונה ההקרנות.
    projection_fps=(number_of_projections/projection_elapsed if projection_elapsed>0 else 0.0)
    # חישוב מספר הפיקסלים של ההקרנות התקינות שעובדו בשנייה.
    pixel_throughput=(total_pixels/projection_elapsed if projection_elapsed>0 else 0.0)
    # הצגת סיכום ביצועי חישוב ההקרנות.
    print("\n--- Projection Performance ---")
    print(f"Complete projections: {number_of_projections}")  # הצגת מספר ההקרנות התקינות שהושלמו.
    print(f"Total projected pixels: {total_pixels:,}")  # הצגת מספר הפיקסלים הכולל של ההקרנות.
    print(f"Projection computation time: {projection_elapsed:.4f} seconds")  # הצגת זמן חישוב ההקרנות.
    print(f"Projection FPS: {projection_fps:.2f}")  # הצגת קצב ההקרנות הממוצע.
    print(f"Projected pixels per second: {pixel_throughput:,.0f}")  # הצגת תפוקת הפיקסלים לשנייה.
    # שלב 2: יצירת הסרטון מתוך כל הפריימים, כולל פריימי שגיאה.
    video_fps=60  # קביעת קצב ההצגה של הסרטון ל-60 פריימים בשנייה.
    video_start=time.perf_counter()  # תחילת מדידת זמן יצירת הסרטון.
    # יצירת הסרטון מתוך רשימת כל הפריימים לפי סדר המישורים המקורי.
    create_video(video_frames, args.output_file, fps=video_fps)
    video_elapsed=time.perf_counter()-video_start  # חישוב זמן כתיבת הסרטון.
    # הצגת סיכום זמני הביצוע וקצבי הפריימים.
    print("\n--- Overall Performance ---")
    print(f"Video writing time: {video_elapsed:.4f} seconds")  # הצגת זמן כתיבת הסרטון.
    print(f"Projection calculation FPS: {projection_fps:.2f}")  # הצגת קצב חישוב ההקרנות התקינות.
    print(f"Video playback FPS: {video_fps}")  # הצגת קצב ההצגה של הסרטון.
    # בדיקה אם ביצועי ההקרנה עומדים ביעדים שהוגדרו במטלה.
    print("\n--- Assignment Performance Check ---")
    if projection_fps>=100:  # בדיקה אם קצב ההקרנות מגיע ליעד הרצוי של 100 הקרנות לשנייה.
        print("Projection performance meets the 100 FPS target.")  # הצגת הודעה שהיעד הרצוי הושג.
    elif projection_fps>=30:  # בדיקה אם קצב ההקרנות מגיע לפחות ל-30 הקרנות לשנייה.
        print("Projection performance meets the 30 FPS minimum target.")  # הצגת הודעה שהיעד המינימלי הושג.
    else:  # הקצב נמוך מהיעד המינימלי.
        print("Projection performance is below the 30 FPS minimum target.\n")  # הצגת הודעה שהיעד המינימלי לא הושג.


# בדיקה אם הקובץ הופעל ישירות ולא יובא כמודול מקובץ Python אחר.
if __name__ == "__main__":
    main()  # הפעלת הפונקציה הראשית שמנהלת את כל שלבי התוכנית.