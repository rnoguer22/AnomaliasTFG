<?php
// Añadimos esto antes de cualquier html para validar el login
session_start();
ini_set('display_errors', 1);
error_reporting(E_ALL);
// Esto es para saltar errores de formato cuando hagamos una inyeccion sql
mysqli_report(MYSQLI_REPORT_OFF);

// Conexion a la base de datos
$conexion = new mysqli("localhost", "admin_tienda", "admin_tfg_uax", "tfg_2026");

// Establecemos la logica del login
$error_login = "";
if (isset($_POST['username']) && isset($_POST['password'])) {
    $user = $_POST['username'];
    $pass = $_POST['password'];
    
    // Hacemos la consulta a la base de datos
    // Es importante que esta consulta sea vulnerable, para poder hacer inyecciones de codigo
    $sql_auth = "SELECT * FROM users WHERE username = '$user' AND password = '$pass'";
    $res_auth = $conexion->query($sql_auth);

    if ($res_auth->num_rows > 0) {
        $_SESSION['user'] = $user;
    } else {
        $error_login = "Inicio de sesión fallido. Credenciales incorrectas.";
    }
}
?>

<!DOCTYPE html>

<html lang="es">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberStore | Entorno de Pruebas TFG</title>
    <link rel="stylesheet" href="style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
</head>
<body>

    <header class="navbar">
        <div class="logo">Cable<span>Store</span></div>
        
        <div style="display: flex; align-items: center; gap: 20px;">
            <?php if(isset($_SESSION['user'])): ?>
                <span style="color: var(--accent); font-weight: bold;">Hola, <?php echo $_SESSION['user']; ?></span>
            <?php endif; ?>

            <form class="search-form" action="index.php" method="GET">
                <input type="text" name="search" placeholder="Buscar cables..." 
                    value="<?php echo isset($_GET['search']) ? $_GET['search'] : ''; ?>">
                <button type="submit">Buscar</button>
            </form>
        </div>
    </header>

    <main class="container">
        <section class="auth-section">
            <h2>Panel de Acceso</h2>
            
            <?php if($error_login != ""): ?>
                <p style="color: #ff4444; font-weight: bold; background: rgba(255,0,0,0.1); padding: 10px; border-radius: 4px;">
                    <?php echo $error_login; ?>
                </p>
            <?php endif; ?>

            <p>Inicia sesión para gestionar tu cuenta.</p>
            
            <form class="login-form" action="index.php" method="POST">
                <div class="input-group">
                    <label for="username">Usuario</label>
                    <input type="text" id="username" name="username" placeholder="admin" required>
                </div>
                <div class="input-group">
                    <label for="password">Contraseña</label>
                    <input type="password" id="password" name="password" placeholder="••••••••" required>
                </div>
                <button type="submit" class="btn-primary">Entrar</button>
                <div class="register-link">¿No tienes cuenta? <a href="#">Regístrate aquí</a></div>
            </form>
        </section>

        <section class="inventory-section">
            <h2>Últimos Productos</h2>
                <div class="product-grid">
                    <?php
                    $search = isset($_GET['search']) ? $_GET['search'] : '';
                    
                    // Hacemos la consulta a la base de datos que nos devuelve los productos
                    // Esto tambien es vulnerable
                    $sql = "SELECT name, description, price FROM products WHERE name LIKE '%$search%'";
                    $resultado = $conexion->query($sql);

                    if ($resultado && $resultado->num_rows > 0) {
                        while($fila = $resultado->fetch_assoc()) {
                            echo '<div class="product-card">';
                            echo '    <div class="product-img-placeholder">CABLE</div>';
                            echo '    <h3>' . $fila["name"] . '</h3>';
                            echo '    <p style="font-size: 0.85rem; color: #94a3b8; margin: 0.5rem 0;">' . $fila["description"] . '</p>';
                            echo '    <p class="price">' . $fila["price"] . ' €</p>';
                            echo '    <button class="btn-secondary">Añadir al carrito</button>';
                            echo '</div>';
                        }
                    } else {
                        echo "<p>No se encontraron productos para: <strong>" . htmlspecialchars($search) . "</strong></p>";
                    }
                    ?>
                </div>
        </section>
    </main>

    <footer>
        <p>Del Bit al Comportamiento: Redefiniendo la frontera de la seguridad mediante el Aprendizaje Profundo</p>
    </footer>

</body>
</html>