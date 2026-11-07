
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
    w000=(1-dx)*(1-dy)*(1-dz)  # חישוב משקל הפינה הראשונה.  # חישוב המשקלים של שמונת הווקסלים המקיפים את נקודת הדגימה.
    w001=(1-dx)*(1-dy)*dz  # חישוב משקל הפינה השנייה.
    w010=(1-dx)*dy*(1-dz)  # חישוב משקל הפינה השלישית.
    w011=(1-dx)*dy*dz  # חישוב משקל הפינה הרביעית.
    w100=dx*(1-dy)*(1-dz)  # חישוב משקל הפינה החמישית.
    w101=dx*(1-dy)*dz  # חישוב משקל הפינה השישית.
    w110=dx*dy*(1-dz)  # חישוב משקל הפינה השביעית.
    w111=dx*dy*dz  # חישוב משקל הפינה השמינית.
    c000=get_voxel(M,x0,y0,z0)  # קריאת ערך הווקסל הראשון.  # קריאת ערכי שמונת הווקסלים מהנפח.
    c001=get_voxel(M,x0,y0,z1)  # קריאת ערך הווקסל השני.
    c010=get_voxel(M,x0,y1,z0)  # קריאת ערך הווקסל השלישי.
    c011=get_voxel(M,x0,y1,z1)  # קריאת ערך הווקסל הרביעי.
    c100=get_voxel(M,x1,y0,z0)  # קריאת ערך הווקסל החמישי.
    c101=get_voxel(M,x1,y0,z1)  # קריאת ערך הווקסל השישי.
    c110=get_voxel(M,x1,y1,z0)  # קריאת ערך הווקסל השביעי.
    c111=get_voxel(M,x1,y1,z1)  # קריאת ערך הווקסל השמיני.
    return (w000*c000+w001*c001+w010*c010+w011*c011+w100*c100+w101*c101+w110*c110+w111*c111)  # חישוב הערך המשוער באמצעות סכום ערכי הווקסלים המשוקללים.


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
    if is_four_points:  # בדיקה אם המישור מוגדר באמצעות ארבע נקודות.  # קביעת ממדי תמונת ההקרנה בהתאם למספר הנקודות.
        H=max(1,round(dist(a,c)))  # קביעת גובה התמונה לפי המרחק בין A ל-C.
        W=max(1,round(dist(c,d)))  # קביעת רוחב התמונה לפי המרחק בין C ל-D.
    else:  # המישור מוגדר באמצעות שלוש נקודות.
        H=max(1,round(dist(a,b)))  # קביעת גובה התמונה לפי המרחק בין A ל-B.
        W=max(1,round(dist(b,c)))  # קביעת רוחב התמונה לפי המרחק בין B ל-C.
    E=np.zeros((H,W),dtype=np.float64)  # יצירת מטריצת הקרנה בגודל H על W, המאותחלת באפס.
    for j in prange(H):  # מעבר מקבילי על שורות תמונת ההקרנה.
        u=(j+0.5)/H  # חישוב מיקום יחסי לאורך הגובה, במרכז כל מקטע דגימה.
        if is_four_points:  # בחירת הגאומטריה המתאימה למישור בעל ארבע נקודות.  # חישוב שתי נקודות הקצה של השורה הנוכחית.
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
            if (px<0 or px>=sx or py<0 or py>=sy or pz<0 or pz>=sz):  # בדיקה אם נקודת הדגימה נמצאת בתוך גבולות הנפח.
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
    בדיקת תקינות המבנה וסוג הנתונים של הנפח התלת־ממדי.

    קלט:
    M - מערך numpy המייצג נפח תלת־ממדי.

    פלט:
    אין. אם הנפח אינו תקין, תיזרק שגיאת ValueError.

    אופן הפעולה:
       בדיקה שהמערך תלת־ממדי, שסוג הנתונים הוא מספר שלם (signed או unsigned),
    ושכל אחד ממדי הנפח נמצא בטווח המותר.
    """
    if M.ndim!=3:  # בדיקה שהמערך הוא תלת־ממדי.
        raise ValueError("The input volume must be a 3D matrix.")  # העלאת שגיאה אם המערך אינו תלת־ממדי.
    if M.dtype not in (np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32):  # בדיקה שסוג הנתונים הוא אחד מסוגי המספרים השלמים הנתמכים.
        raise ValueError("Unsupported data type.")  # העלאת שגיאה אם סוג הנתונים אינו נתמך.
    sx,sy,sz=M.shape  # קבלת ממדי הנפח.
    if not (1<=sx<2**10 and 1<=sy<2**10 and 1<=sz<2**10):  # בדיקה שכל ממד נמצא בטווח המותר.
        raise ValueError("Volume dimensions must be between 1 and 1023.")  # העלאת שגיאה אם אחד הממדים אינו תקין.

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
        a,b,c=[np.asarray(p, dtype=float) for p in points]  # המרת הנקודות למערכי NumPy לצורך חישובים וקטוריים
        dab=dist(a,b)  # המרחק בין A ל-B  # חישוב המרחקים בין כל זוג נקודות
        dac=dist(a,c)  # המרחק בין A ל-C
        dbc=dist(b,c)  # המרחק בין B ל-C
        print(f"  dab = {dab:.6f}, dac = {dac:.6f}, dbc = {dbc:.6f}")
        print(f"  |dab - dac| = {abs(dab - dac):.6f}")
        valid_equal_sides=abs(dab-dac)<EPS  # בדיקה שהמרחקים AB ו-AC שווים בקירוב לפי EPS
        valid_lengths=dab>0 and dac>0 and dbc>0  # בדיקה שכל המרחקים בין הנקודות חיוביים
        cross_product=np.cross(b-a,c-a)  # חישוב המכפלה הווקטורית של AB ו-AC אם שלוש הנקודות נמצאות על אותו קו, גודל המכפלה יהיה אפס
        parallelogram_area=np.linalg.norm(cross_product)  # חישוב שטח המקבילית שנוצרת על ידי שני הווקטורים
        longest_side=max(dab,dac,dbc)  # חישוב אורך הצלע הארוכה ביותר לצורך קבלת גובה גאומטרי
        valid_non_collinear =(longest_side>0 and parallelogram_area/longest_side>EPS)  # חלוקת שטח המקבילית באורך הצלע הארוכה נותנת גובה כך ניתן להשוות את הגובה לסף EPS, שהוא סף של מרחק
        print(f"  Equal sides: {valid_equal_sides}")  # הדפסת תוצאות הבדיקות לצורך איתור שגיאות
        print(f"  Positive lengths: {valid_lengths}")
        print(f"  Non-collinear points: {valid_non_collinear}")
        return (valid_equal_sides and valid_lengths and valid_non_collinear)  # המישור תקין רק אם כל התנאים מתקיימים
    if len(points)==4:  # בדיקה אם המישור מוגדר באמצעות ארבע נקודות.
        a1,a2,b,c=[np.asarray(p,dtype=float) for p in points]  # המרת ארבע הנקודות למערכים מספריים.
        side1=dist(a1,b)  # חישוב אורך הצלע הראשונה.
        side2=dist(a2,c)  # חישוב אורך הצלע השנייה.
        base1=dist(a1,a2)  # חישוב אורך הבסיס הראשון.
        base2=dist(b,c)  # חישוב אורך הבסיס השני.
        normal=np.cross(a2-a1,b-a1)  # חישוב וקטור הנורמל למישור שמוגדר על ידי a1, a2 ו-b.
        normal_length=np.linalg.norm(normal)  # חישוב אורך וקטור הנורמל.
        if normal_length==0:  # בדיקה שהנקודות a1, a2 ו-b אינן נמצאות על אותו קו.
            coplanarity=float("inf")  # אם אין מישור מוגדר, הבדיקה נכשלת.
        else:
            coplanarity=abs(np.dot(normal,c-a1))/normal_length  # חישוב המרחק של הנקודה c מהמישור.
        print(f"  side1 = {side1:.6f}, side2 = {side2:.6f}")  # הצגת אורכי הצלעות.
        print(f"  base1 = {base1:.6f}, base2 = {base2:.6f}")  # הצגת אורכי הבסיסים.
        print(f"  |side1 - side2| = {abs(side1 - side2):.6f}")  # הצגת ההפרש בין אורכי הצלעות.
        print(f"  Coplanarity distance = {coplanarity:.6f}")  # הצגת המרחק של הנקודה c מהמישור.
        valid_equal_sides=abs(side1-side2)<EPS  # בדיקה אם אורכי הצלעות שווים בקירוב.
        valid_coplanarity=coplanarity<EPS  # בדיקה אם ארבע הנקודות נמצאות באותו מישור בקירוב.
        valid_bases=base1<=base2  # בדיקה שסדר אורכי הבסיסים עומד בתנאי הקוד.
        valid_lengths=(side1>0 and side2>0 and base1>0 and base2>0)  # בדיקה שכל האורכים חיוביים.
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
       רשימת נקודות בסדר תקין. אם לא נמצא סדר מתאים, תיזרק שגיאת ValueError.

    אופן הפעולה:
    תחילה נבדק הסדר המקורי. אם הוא אינו תקין, הפונקציה
    עוברת על תמורות הנקודות ובודקת כל סידור עד למציאת סידור תקין.
    """
    if validate_plane_geometry(points):  # בדיקה אם הסדר המקורי תקין.  # בדיקת הסדר המקורי לפני חיפוש סידורים אחרים.
        return points  # החזרת הרשימה המקורית אם היא תקינה.
    for candidate in permutations(points):  # יצירת התמורות האפשריות של הנקודות.  # חיפוש סידור חלופי רק אם הסדר המקורי אינו תקין.
        candidate=list(candidate)  # המרת התמורה לרשימה.
        if candidate==points:  # בדיקה אם התמורה זהה לרשימה המקורית שכבר נבדקה.
            continue  # דילוג לתמורה הבאה.
        if validate_plane_geometry(candidate):  # בדיקה אם הסידור החדש תקין.
            return candidate  # החזרת הסידור הראשון שעבר את הבדיקה.
    raise ValueError("Invalid plane geometry: no valid point ordering found.")  # העלאת שגיאה אם לא נמצא סידור תקין של נקודות המישור.


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



def create_error_frame(plane_number, points, error_message=None):
    """
    מטרת הפונקציה:
    ליצור פריים שגיאה עבור מישור שאינו תקין.

    קלט:
    plane_number - מספר המישור לפי סדרו בקובץ CSV.
    points - הנקודות המקוריות שהתקבלו מקובץ CSV.
    error_message - הודעת השגיאה, אם נמצאה שגיאה בקריאת שורת ה-CSV.

    פלט:
    תמונת שגיאה בגווני אפור הכוללת את מספר המישור ואת פרטי השגיאה.

    אופן הפעולה:
    נוצרת תמונה שחורה בגודל קבוע.
    על התמונה נכתבת כותרת המציינת את מספר המישור ואת העובדה שהוא אינו תקין.
    אם התקבלה הודעת שגיאה, היא מוצגת על גבי התמונה.
    לאחר מכן מוצגות הקואורדינטות של הנקודות התקינות שהתקבלו, אם קיימות.
    """
    frame=np.zeros((420,700),dtype=np.uint8)  # יצירת תמונה שחורה בגודל 700 על 420 פיקסלים.
    cv2.putText(frame,f"PLANE {plane_number}-ERROR",(25,55),cv2.FONT_HERSHEY_SIMPLEX,1.0,255,2)  # הצגת כותרת השגיאה על גבי התמונה.
    cv2.putText(frame,"INVALID PLANE",(25,105),cv2.FONT_HERSHEY_SIMPLEX,0.8,255,2)  # הצגת הודעה המציינת שהמישור אינו תקין.
    start_y=165  # קביעת מיקום התחלתי להצגת הקואורדינטות.
    if error_message is not None:  # בדיקה אם התקבלה הודעת שגיאה מקריאת שורת ה-CSV.
        cv2.putText(frame,f"CSV ERROR: {error_message[:65]}",(25,155),cv2.FONT_HERSHEY_SIMPLEX,0.45,255,1)  # הצגת הודעת השגיאה על גבי הפריים.
        start_y=205  # הזזת הקואורדינטות כלפי מטה כדי שלא יחפפו להודעת השגיאה.
    for i,point in enumerate(points):  # הצגת כל הנקודות המקוריות שהתקבלו עבור המישור.
        point_text=(f"P{i+1} = "f"({point[0]:.3f},{point[1]:.3f},{point[2]:.3f})")  # בניית מחרוזת הכוללת את מספר הנקודה ואת הקואורדינטות שלה.
        cv2.putText(frame,point_text,(25,start_y+i*45),cv2.FONT_HERSHEY_SIMPLEX,0.65,255,2)  # כתיבת הקואורדינטות על גבי פריים השגיאה.
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
    max_height=max(p.shape[0] for p in projections)  # מציאת גובה הפריים הגדול ביותר.  # קביעת הגודל האחיד של הסרטון לפי הפריים הגדול ביותר ברשימה.
    max_width=max(p.shape[1] for p in projections)  # מציאת רוחב הפריים הגדול ביותר.
    video_height=max_height+(max_height%2)  # התאמת גובה הסרטון למספר זוגי.  # התאמת הממדים למספרים זוגיים, כנדרש על ידי קודקי MP4 נפוצים.
    video_width=max_width+(max_width%2)  # התאמת רוחב הסרטון למספר זוגי.
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")  # בחירת קודק לקידוד הסרטון בפורמט MP4V.
    writer=cv2.VideoWriter(output_file,fourcc,fps,(video_width,video_height),isColor=True)  # יצירת כותב הווידאו עם הנתיב, הקודק, קצב הפריימים וגודל הסרטון.
    if not writer.isOpened():  # בדיקה אם כותב הווידאו נפתח בהצלחה.
        raise RuntimeError(f"Could not create video: {output_file}")  # העלאת שגיאה אם לא ניתן ליצור את הקובץ.
    total_frames=0  # אתחול מונה הפריימים שנכתבו לסרטון.
    try:  # התחלת בלוק שמבטיח שחרור משאבים גם במקרה של שגיאה.
        for plane_number,projection in enumerate(projections,start=1):  # מעבר על כל הפריימים לפי הסדר שבו הם מופיעים ברשימה.
            height,width=projection.shape  # קבלת גובה ורוחב הפריים הנוכחי.
            frame=np.zeros((video_height,video_width),dtype=np.uint8)  # יצירת פריים שחור בגודל האחיד של הסרטון.
            image=np.clip(np.rint(projection), 0, 255).astype(np.uint8)  # עיגול ערכי הפריים, הגבלתם לטווח 0–255 והמרתם ל-8 ביט.
            y_offset=(video_height-height)//2  # חישוב ההיסט האנכי למרכוז התמונה.
            x_offset=(video_width-width)//2  # חישוב ההיסט האופקי למרכוז התמונה.
            frame[y_offset:y_offset+height,x_offset:x_offset+width]=image  # הצבת התמונה המלאה במרכז הפריים השחור.
            color_frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)  # המרת הפריים מגווני אפור לפורמט BGR בעל שלושה ערוצים.
            writer.write(color_frame)  # כתיבת הפריים לקובץ הווידאו.
            total_frames+=1  # הגדלת מונה הפריימים לאחר כתיבת פריים אחד.
            print(f"Plane {plane_number}/{len(projections)} " f"written; total frames: {total_frames}")  # הצגת התקדמות כתיבת הפריימים לסרטון.
    finally:  # בלוק שמתבצע גם במקרה של שגיאה בזמן כתיבת הסרטון.
        writer.release()  # שחרור כותב הווידאו וסגירת הקובץ.
    print(f"Video saved to: {output_file}")  # הצגת נתיב קובץ הווידאו.  # הצגת פרטי הסרטון שנוצר.
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
    2. בדיקת תקינות הנפח והמרת סוג הנתונים במידת הצורך.
    3. קריאת המישורים מקובץ CSV.
    4. בדיקת תקינות גאומטרית וסידור הנקודות.
    5. יצירת פריימי שגיאה עבור מישורים לא תקינים.
    6. חישוב ההקרנות התקינות ומדידת זמני הביצוע.
    7. יצירת סרטון MP4 לפי סדר המישורים המקורי.
    8. הצגת סיכום הביצועים ובדיקת יעדי הקצב.
    """
    import argparse  # ייבוא הספרייה לקריאת ארגומנטים משורת הפקודה.
    parser=argparse.ArgumentParser(description="Project planes onto a 3D volume and create an MP4 video.")  # יצירת מנתח הארגומנטים של התוכנית.
    parser.add_argument("nrrd_file",help="Input NRRD volume")  # הגדרת הארגומנט של קובץ הנפח.
    parser.add_argument("csv_file",help="CSV file containing plane definitions")  # הגדרת הארגומנט של קובץ המישורים.
    parser.add_argument("output_file",help="Output MP4 video path")  # הגדרת הארגומנט של קובץ הפלט.
    args=parser.parse_args()  # קריאת הארגומנטים שהועברו בעת הפעלת התוכנית.
    M,header=nrrd.read(args.nrrd_file, index_order="C")  # טעינת הנפח התלת־ממדי והמטא־נתונים מקובץ NRRD.
    if M.size==0:  # בדיקה אם המערך ריק.
        raise ValueError("The input volume is empty.")  # הצגת שגיאה אם הנפח ריק.
    validate_volume(M)  # בדיקת תקינות מבנה הנפח, ממדיו וסוג הנתונים לפני ביצוע ההמרה.
    try:  # ניסיון לבדוק ולהמיר את הנפח לסוג הנתונים uint16.
        if not np.issubdtype(M.dtype, np.integer):  # בדיקה אם הנתונים הם מספרים שלמים.
            raise ValueError("The input volume must contain integer values.")  # העלאת שגיאה אם הנתונים אינם מספרים שלמים.
        if M.min()<0 or M.max()>65535:  # בדיקה שכל ערכי הנפח נמצאים בטווח המותר של uint16.
            raise ValueError("Volume values must be between 0 and 65535.")  # העלאת שגיאה אם הערכים מחוץ לטווח.
        M=M.astype(np.uint16)  # המרת הנפח לסוג הנתונים uint16.
    except (TypeError,ValueError,OverflowError) as error:  # תפיסת שגיאות שעלולות להתרחש במהלך הבדיקה או ההמרה.
        raise ValueError(f"Cannot convert volume to uint16: {error}") from error  # העלאת שגיאה עם הסבר על כישלון ההמרה.
    print("Volume shape:", M.shape)  # הצגת מידע על הנפח ועל מערכת הקואורדינטות שלו.
    print("NRRD space:", header.get("space"))
    print("Space origin:", header.get("space origin"))
    print("Space directions:\n", header.get("space directions"))
    print("Volume dtype:", M.dtype)
    print("Volume value range:", M.min(), M.max())
    print("CSV plane coordinates are interpreted as voxel indices (X, Y, Z).")  # ציון האופן שבו מתפרשות הקואורדינטות בקובץ CSV.
    
    planes=[]  # יצירת רשימה לאחסון הגדרות המישורים מקובץ ה-CSV.
    with open(args.csv_file, newline="",encoding="utf-8-sig") as csvfile:  # פתיחת קובץ ה-CSV לקריאה, תוך תמיכה בקידוד UTF-8 עם BOM.
        reader=csv.reader(csvfile)  # יצירת אובייקט לקריאת שורות הקובץ.
        next(reader,None)  # דילוג על שורת הכותרת של קובץ ה-CSV.
        plane_number=0  # אתחול מונה המישורים שאינם שורות הערה או שורות ריקות.
        for row_number,row in enumerate(reader,start=2):  # מעבר על שורות הקובץ, כאשר מספר השורה מתחיל מ-2 בגלל הכותרת.
            if not row or all(not value.strip() for value in row):  # בדיקה אם השורה ריקה או מכילה רווחים בלבד.
                continue  # דילוג לשורה הבאה.
            if row[0].strip().startswith("#"):  # בדיקה אם השורה מכילה הערה שמתחילה בסימן #.
                continue  # דילוג על שורת ההערה.
            plane_number+=1  # הגדלת מונה המישורים עבור שורת נתונים.
            try:  # התחלת טיפול בשגיאות שעלולות להתרחש בעת קריאת השורה.
                num_points=int(row[0])  # קריאת מספר הנקודות המוגדרות בשורה.
                if num_points not in (3,4):  # בדיקה שמספר הנקודות הוא שלוש או ארבע.
                    raise ValueError("A plane must contain 3 or 4 points.")  # העלאת שגיאה אם מספר הנקודות אינו נתמך.
                expected_values=1+num_points*3  # חישוב מספר השדות הצפוי בשורה לפי מספר הנקודות.
                while row and not row[-1].strip():  # הסרת שדות ריקים שמופיעים בסוף שורת ה-CSV.
                    row.pop()
                if len(row)!=expected_values:  # בדיקה שמספר השדות לאחר הסרת השדות הריקים תקין.
                    raise ValueError("Incorrect number of coordinate values.")  # העלאת שגיאה אם מספר השדות אינו תקין.
                values=[float(value) for value in row[1:]]  # קריאת הקואורדינטות והמרתן למספרים עשרוניים.
                if not all(math.isfinite(value) for value in values):  # בדיקה שכל הקואורדינטות הן מספרים סופיים.
                    raise ValueError("Coordinates must be finite numbers.")  # העלאת שגיאה אם נמצאו ערכים שאינם סופיים.
                points=[tuple(values[i:i+3]) for i in range(0,len(values),3)]  # חלוקת הקואורדינטות לקבוצות של שלושה ערכים: X, Y, Z.
                planes.append((points,None))  # שמירת הנקודות התקינות ללא הודעת שגיאה.
            except (ValueError,IndexError) as error:  # תפיסת שגיאות המרה או גישה לאיברים בשורה.
                print(f"ERROR: CSV row {row_number}, plane {plane_number}: {error}")  # הצגת מספר השורה והמישור שבהם נמצאה השגיאה.
                planes.append((None,str(error)))  # שמירת המישור הפגום כדי ליצור עבורו פריים שגיאה ולהמשיך לקרוא את הקובץ.
    if not planes:  # בדיקה אם לא נמצאו הגדרות מישורים בקובץ.
        raise ValueError("The CSV file contains no plane definitions.")  # עצירת התוכנית אם אין מישורים.
    plane_jobs=[]  # יצירת רשימת משימות שתשמור על הסדר המקורי של המישורים בקובץ CSV.
    for i,(points,row_error) in enumerate(planes,start=1):  # מעבר על כל המישורים והפרדת הנקודות מהודעות השגיאה.
        if row_error is not None:  # בדיקה אם נמצאה שגיאה בעת קריאת שורת ה-CSV.
            print(f"ERROR: Plane {i} is invalid: {row_error}")  # הצגת הודעת השגיאה בטרמינל.
            error_frame=create_error_frame(i,[],row_error)  # יצירת פריים שגיאה המציג גם את סיבת השגיאה בשורת ה-CSV.
            plane_jobs.append(("error",error_frame))  # שמירת פריים השגיאה במיקום המתאים בסרטון.
            continue  # מעבר למישור הבא ללא בדיקה גאומטרית.
        try:  # ניסיון למצוא סדר תקין לנקודות המישור.
            ordered_points=find_valid_plane_order(points)  # בדיקת הסדר המקורי וחיפוש סדר חלופי אם הסדר המקורי אינו תקין.
        except ValueError as error:  # תפיסת שגיאה במקרה שלא נמצא סידור תקין של נקודות המישור.
            print(f"ERROR: Plane {i} is invalid: {error}")  # הצגת הודעת שגיאה בטרמינל.
            error_frame=create_error_frame(i,points)  # יצירת פריים שגיאה המציג את מספר המישור ואת הקואורדינטות המקוריות.
            plane_jobs.append(("error",error_frame))  # שמירת פריים השגיאה במיקום המתאים ברצף הסרטון.
        else:  # נמצא סדר נקודות תקין עבור המישור.
            if ordered_points!=points:  # בדיקה אם היה צורך לשנות את סדר הנקודות.
                print(f"Plane {i}: valid - point order updated")
            else:  # הסדר המקורי נמצא תקין.
                print(f"Plane {i}: valid")
            plane_jobs.append(("valid",ordered_points))  # שמירת הנקודות התקינות לצורך חישוב ההקרנה בהמשך.
    dummy_3_points=[(-12.75,8.35,-3.5),(4.25,-16.8,7.65),(-9.4,-5.25,13.2)]  # שלב 1: חישוב ההקרנות התקינות ומדידת הביצועים שלהן. חימום Numba לפני מדידת זמן חישוב ההקרנות. מטרת החימום היא לבצע קומפילציה של פונקציות ההקרנה לפני המדידה. כך זמן הקומפילציה הראשוני לא ייכלל בזמן הביצוע הנמדד. יצירת נקודות מלאכותיות עבור מישור המוגדר באמצעות 3 נקודות.
    dummy_4_points=[(-12.75,8.35,-3.5),(4.25,-16.8,7.65),(-9.4,-5.25,13.2),(15.6,-11.45,-8.25)]  # יצירת נקודות מלאכותיות עבור מישור המוגדר באמצעות 4 נקודות.
    _ = project_plane_3(M,*dummy_3_points)  # חימום פונקציית ההקרנה עבור שלוש נקודות.  # הפעלת שתי פונקציות ההקרנה לצורך קומפילציה מראש.
    _ = project_plane_4(M,*dummy_4_points)  # חימום פונקציית ההקרנה עבור ארבע נקודות.
    print("Numba warm-up completed for both plane types.")  # הצגת הודעה על סיום שלב החימום.
    projection_start=time.perf_counter()  # שמירת זמן ההתחלה למדידת חישוב ההקרנות.  # התחלת מדידת זמן ההקרנות לאחר פעולת החימום.
    projections=[]  # רשימת הקרנות תקינות בלבד, המשמשת לחישוב נתוני הביצועים.
    video_frames=[]  # רשימת כל הפריימים, כולל הקרנות ופריימי שגיאה, לפי הסדר המקורי.
    total_pixels=0  # אתחול מונה הפיקסלים של ההקרנות התקינות בלבד.
    for plane_number,(job_type,data) in enumerate(plane_jobs,start=1):  # מעבר על כל משימות המישורים לפי הסדר המקורי.
        if job_type=="error":  # בדיקה אם המשימה מכילה פריים שגיאה.
            video_frames.append(data)  # הוספת פריים השגיאה למיקומו המקורי בסרטון.
            continue  # מעבר למישור הבא ללא חישוב הקרנה וללא הכללתו במדידת FPS.
        points=data  # שמירת נקודות המישור התקין במשתנה נוח לשימוש.
        plane_start=time.perf_counter()  # שמירת זמן ההתחלה של חישוב ההקרנה הנוכחית.
        projection=project_valid_plane(M,points)  # חישוב ההקרנה הדו־ממדית של המישור התקין.
        plane_elapsed=time.perf_counter()-plane_start  # חישוב זמן ההקרנה של המישור הנוכחי.
        projections.append(projection)  # שמירת ההקרנה ברשימת ההקרנות התקינות וברשימת פריימי הסרטון.
        video_frames.append(projection)
        height, width=projection.shape  # קבלת גובה ורוחב תמונת ההקרנה.
        total_pixels+=height*width  # הוספת מספר הפיקסלים של ההקרנה למונה הכולל.
        plane_fps=1.0/plane_elapsed if plane_elapsed>0 else 0.0  # חישוב קצב ההקרנות של המישור הנוכחי.
        print(f"Plane {plane_number}: {height} rows x {width} columns")  # הצגת ממדי ההקרנה וזמן החישוב שלה.
        print(f"  Projection time: {plane_elapsed:.6f} seconds")
        print(f"  Projection FPS: {plane_fps:.2f}")
    projection_elapsed=time.perf_counter()-projection_start  # חישוב זמן הטיפול הכולל במשימות ההקרנה. זמן זה כולל את מעבר הלולאה ואת הטיפול בפריימי שגיאה.
    number_of_projections=len(projections)  # ספירת מספר ההקרנות התקינות שחושבו.
    projection_fps=(number_of_projections/projection_elapsed if projection_elapsed>0 else 0.0)  # חישוב מספר ההקרנות התקינות לשנייה. פריימי שגיאה אינם נכללים במונה ההקרנות.
    pixel_throughput=(total_pixels/projection_elapsed if projection_elapsed>0 else 0.0)  # חישוב מספר הפיקסלים של ההקרנות התקינות שעובדו בשנייה.
    print("\n--- Projection Performance ---")  # הצגת סיכום ביצועי חישוב ההקרנות.
    print(f"Complete projections: {number_of_projections}")  # הצגת מספר ההקרנות התקינות שהושלמו.
    print(f"Total projected pixels: {total_pixels:,}")  # הצגת מספר הפיקסלים הכולל של ההקרנות.
    print(f"Projection computation time: {projection_elapsed:.4f} seconds")  # הצגת זמן חישוב ההקרנות.
    print(f"Projection FPS: {projection_fps:.2f}")  # הצגת קצב ההקרנות הממוצע.
    print(f"Projected pixels per second: {pixel_throughput:,.0f}")  # הצגת תפוקת הפיקסלים לשנייה.
    video_fps=60  # קביעת קצב ההצגה של הסרטון ל-60 פריימים בשנייה.  # שלב 2: יצירת הסרטון מתוך כל הפריימים, כולל פריימי שגיאה.
    video_start=time.perf_counter()  # תחילת מדידת זמן יצירת הסרטון.
    create_video(video_frames, args.output_file, fps=video_fps)  # יצירת הסרטון מתוך רשימת כל הפריימים לפי סדר המישורים המקורי.
    video_elapsed=time.perf_counter()-video_start  # חישוב זמן כתיבת הסרטון.
    print("\n--- Overall Performance ---")  # הצגת סיכום זמני הביצוע וקצבי הפריימים.
    print(f"Video writing time: {video_elapsed:.4f} seconds")  # הצגת זמן כתיבת הסרטון.
    print(f"Projection calculation FPS: {projection_fps:.2f}")  # הצגת קצב חישוב ההקרנות התקינות.
    print(f"Video playback FPS: {video_fps}")  # הצגת קצב ההצגה של הסרטון.
    print("\n--- Assignment Performance Check ---")  # בדיקה אם ביצועי ההקרנה עומדים ביעדים שהוגדרו במטלה.
    if projection_fps>=100:  # בדיקה אם קצב ההקרנות מגיע ליעד הרצוי של 100 הקרנות לשנייה.
        print("Projection performance meets the 100 FPS target.")  # הצגת הודעה שהיעד הרצוי הושג.
    elif projection_fps>=30:  # בדיקה אם קצב ההקרנות מגיע לפחות ל-30 הקרנות לשנייה.
        print("Projection performance meets the 30 FPS minimum target.")  # הצגת הודעה שהיעד המינימלי הושג.
    else:  # הקצב נמוך מהיעד המינימלי.
        print("Projection performance is below the 30 FPS minimum target.\n")  # הצגת הודעה שהיעד המינימלי לא הושג.


if __name__ == "__main__":  # בדיקה אם הקובץ הופעל ישירות ולא יובא כמודול מקובץ Python אחר.
    main()  # הפעלת הפונקציה הראשית שמנהלת את כל שלבי התוכנית.