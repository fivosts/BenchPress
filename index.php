<head>
  <title>CLGEN samples</title>
</head>
<body style = "background-color:#272822;">

  <h1 style = "color:white;">Write here the model specs ?</h1>
  <br>
  <h2 style = "color:white;">Write here if it is validation or sampling</h2>

  <h2 style = "color:white;">Available workspaces:</h2>


  <p style = "color:white;">

  <?php

  $a = shell_exec("python test.py workspaces");

  echo $a;

  ?>

<?php

function list_models($base_path)
{
  // echo "<br>";
  $dh = opendir($base_path);
  $i=1;
  while (($file = readdir($dh)) !== false) {
      if($file != "." && $file != ".." && $file != "index.php" && $file != ".htaccess" && $file != "error_log" && $file != "cgi-bin" && is_dir($base_path . "/" . $file)) {
          // echo "<a href='$base_path/$file' style=\"color:grey;\" >$file</a><br /><br />";
          echo "$file<br /><br />";
          $curr_path = $base_path . "/" . $file;
          echo file_get_contents($curr_path . "/META.pbtxt");
          $i++;
      }
  }
  closedir($dh);
}

function list_workspace_folders($base_path)
{
  echo "<br>";
  $dh = opendir($base_path);
  $i=1;
  while (($file = readdir($dh)) !== false) {
      if($file != "." && $file != ".." && $file != "index.php" && $file != ".htaccess" && $file != "error_log" && $file != "cgi-bin" && is_dir($base_path . "/" . $file)) {
          echo "<a href='$base_path/$file' style=\"color:grey;\" >$file</a><br /><br />";
          // echo "$file<br /><br />";
          if ($file == "model")
          {
            $curr_path = $base_path . "/" . $file;
            list_models($curr_path);
          }
          $i++;
      }
  }
  closedir($dh);
}

function list_workspaces($base_path)
{
  echo "<br>";
  $dh = opendir($base_path);
  $i=1;
  while (($file = readdir($dh)) !== false) {
      if($file != "." && $file != ".." && $file != "index.php" && $file != ".htaccess" && $file != "error_log" && $file != "cgi-bin" && is_dir($base_path . "/" . $file)) {
          // echo "<a href='$base_path/$file'>$file</a><br /><br />";
          echo "$file<br /><br />";
          $curr_path = $base_path . "/" . $file;
          list_workspace_folders($curr_path);
          $i++;
      }
  }
  closedir($dh);
}

$workspace_path = "./workspace";
list_workspaces($workspace_path);

?> 

  </p>



</body>
</html>
