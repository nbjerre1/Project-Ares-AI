using Godot;
using System;
using System.Collections.Generic;
using static System.Runtime.InteropServices.JavaScript.JSType;

public partial class Player : CharacterBody2D
{
    [Export] public float Speed = 400f;
    [Export] public float ShootCooldown = 0.5f;
    [Signal] public delegate void XpChangedEventHandler(int newXp);
    [Export] public PackedScene UpgradeMenuScene { get; set; }
    [Export] public bool IsAI = true;
    [Export] public int PlayerId = 1;
    public int DiagonalShotLevel { get; set; } = 0;
    private static readonly float[][] DiagonalShotAngles = new float[][]
{
    new float[] { 45f, -45f },   // Upgrade 1
    new float[] { 40f, -40f },   // Upgrade 2
    new float[] { 50f, -50f },   // Upgrade 3
    // Add more if you want more upgrades
};



    private float shootCooldownTimer = 0f;
    private Sprite2D sprite;
    private static readonly PackedScene ArrowScene = GD.Load<PackedScene>("res://Scenes/Arrow.tscn");
    private HP _health;
    public int Health => _health != null ? _health.CurrentHealth : 0;

    public int _xp = 0;
    public int XpToLevel = 100;
    public PlayerData Data { get; private set; }

    // Track enemies in range
    private List<Node2D> enemiesInRange = new();

    public override void _Ready()
    {
        sprite = GetNode<Sprite2D>("PlayerLars");
        var detectionArea = GetNode<Area2D>("DetectionArea");
        detectionArea.BodyEntered += OnDetectionAreaBodyEntered;
        detectionArea.BodyExited += OnDetectionAreaBodyExited;

        //HP stuff
        _health = GetNode<HP>("Health");
        _health.Died += OnPlayerDied;
        _health.HealthChanged += OnHealthChanged;        

        Data = PlayerData.Load();
    }

    public override void _PhysicsProcess(double delta)
    {
        Vector2 direction = Vector2.Zero;

        if (IsAI)
        {
            string action = SocketServer.LatestActions.TryGetValue(PlayerId, out var a) ? a : "idle";
            SetVelocityFromAction(action);
            GD.Print("Received action from socket: " + action);
            // Optionally, set direction here if you want to use it for flipping, etc.
        }
        else
        {
            direction = Input.GetVector("move_left", "move_right", "move_up", "move_down");
            Velocity = direction * Speed;
        }

        // If not AI, direction is set above; if AI, you may want to set direction based on Velocity
        if (!IsAI)
        {
            Velocity = direction * Speed;
        }
        MoveAndSlide();

        if (direction.X != 0)
            sprite.FlipH = direction.X < 0;

        if (shootCooldownTimer > 0f)
            shootCooldownTimer -= (float)delta;

        // Shoot if any enemy is in range and cooldown is ready
        if (enemiesInRange.Count > 0 && shootCooldownTimer <= 0f)
        {
            Shoot();
            shootCooldownTimer = ShootCooldown;
        }
    }


    private void Shoot()
    {
        if (enemiesInRange.Count == 0)
            return;

        var shootingPoint = GetNode<Node2D>("ShootingPoint");
        if (shootingPoint == null)
            return;

        Node2D targetEnemy = null;
        float minDist = float.MaxValue;

        var spaceState = GetWorld2D().DirectSpaceState;

        foreach (var enemy in enemiesInRange)
        {
            // Raycast from shooting point to enemy
            var query = PhysicsRayQueryParameters2D.Create(
                shootingPoint.GlobalPosition,
                enemy.GlobalPosition
            );
            // Optionally, set collision mask or exclude the player itself
            query.Exclude = new Godot.Collections.Array<Rid> { this.GetRid() };

            var result = spaceState.IntersectRay(query);

            // Check if the first object hit is the enemy
            if (result.Count > 0 && result["collider"].As<Node2D>() != enemy)
                continue; // Something is blocking the shot
           

            float dist = shootingPoint.GlobalPosition.DistanceTo(enemy.GlobalPosition);
            if (dist < minDist)
            {
                minDist = dist;
                targetEnemy = enemy;
            }
        }

        if (targetEnemy == null)
            return;

        Vector2 toEnemy = (targetEnemy.GlobalPosition - shootingPoint.GlobalPosition).Normalized();
        float baseRotation = toEnemy.Angle();

        // Always fire the straight arrow
        SpawnArrow(shootingPoint.GlobalPosition, baseRotation);

        // Fire extra arrows for each diagonal upgrade level
        int diagonalLevel = DiagonalShotLevel;
        for (int i = 0; i < diagonalLevel && i < DiagonalShotAngles.Length; i++)
        {
            foreach (float angleDeg in DiagonalShotAngles[i])
            {
                float angleRad = Mathf.DegToRad(angleDeg);
                SpawnArrow(shootingPoint.GlobalPosition, baseRotation + angleRad);
            }
        }
    }




    private void SpawnArrow(Vector2 position, float rotation)
    {
        var newArrow = ArrowScene.Instantiate();
        if (newArrow is Arrow arrowNode)
        {
            arrowNode.GlobalPosition = position;
            arrowNode.GlobalRotation = rotation;
            arrowNode.playerData = Data;
            GetTree().CurrentScene.AddChild(arrowNode);
        }
    }




    private void OnDetectionAreaBodyEntered(Node2D body)
    {
        if (body.IsInGroup("enemies") && !enemiesInRange.Contains(body))
            enemiesInRange.Add(body);
    }

    private void OnDetectionAreaBodyExited(Node2D body)
    {
        if (body.IsInGroup("enemies"))
            enemiesInRange.Remove(body);
    }
    private void OnHealthChanged(int newHp)
    {
        // Example: update health bar UI, ingen ui endnu !!!!
        var healthBar = GetNode<ProgressBar>("HealthBar");
        healthBar.Value = newHp;
        healthBar.Visible = newHp < _health.MaxHealth;
    }

    private void OnPlayerDied()
    {
        GD.Print("Player died!");
        // Send final state with health = 0 to the socket server
        var state = GetState();
        state["health"] = 0; // Ensure health is 0
        var socketServer = GetNode<SocketServer>("/root/SocketServer");
        if (socketServer != null)
            socketServer.SendStateToAI(state);// <-- Implement this method in your SocketServer

        // Do NOT shut down the socket server!
        GetTree().ReloadCurrentScene(); // Or handle game over
    }

    public void AddXp(int amount)
    {
        _xp += amount;
        GD.Print($"Player gained {amount} XP. Total XP: {_xp}");
       
        EmitSignal(nameof(XpChanged), _xp);
    }

    public void AddGold(int amount)
    {        
        Data.AddGold(amount);         
        Data.Save();
            
        
    }
    
    private void SetVelocityFromAction(string action)
    {
        Vector2 velocity = Vector2.Zero;
        switch (action)
        {
            case "move_left":
                velocity.X = -Speed;
                break;
            case "move_right":
                velocity.X = Speed;
                break;
            case "move_up":
                velocity.Y = -Speed;
                break;
            case "move_down":
                velocity.Y = Speed;
                break;
        }
        Velocity = velocity;
    }
    private void HandleAction(string action)
    {
        Vector2 velocity = Vector2.Zero;

        switch (action)
        {
            case "move_left":
                velocity.X = -Speed;
                break;
            case "move_right":
                velocity.X = Speed;
                break;

        }

        Velocity = velocity;
        MoveAndSlide();
    }

    public Dictionary<string, object> GetState()
    {
        return new Dictionary<string, object>
    {
        { "x", (float)GlobalPosition.X },
        { "y", (float)GlobalPosition.Y },
        { "health", (int)Health },
        { "timer", StopwatchLabel.GameTime },
        { "xpdrop", _xp }
    };
    }

}
